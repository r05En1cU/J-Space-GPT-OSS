#!/usr/bin/env python3
"""Discriminative layer-16 orbit/plant J-Lens diagnostic.

Runs two checks requested for the readout-full JVP path (A) vs per-token
VJP dictionary path (B):
  1) MoE route equality for the actual existing bf16 A/B implementations.
  2) A custom small-case downstream recomputation where layer-16 -> final
     matmul weights are lifted on the fly to fp32/fp64 for both A and B.

This file is intentionally standalone and does not change approved J-space math.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import jspace_gpt_oss as jl  # noqa: E402


TRACK_TOKENS = [" orbit", " plant"]


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def resolve_positions(batch: Mapping[str, torch.Tensor], position_mode: str, max_pairs: int) -> List[Tuple[int, int]]:
    seq_len = int(batch["input_ids"].shape[1])
    return jl.make_position_pairs(seq_len, position_mode, max_pairs)


def capture_layer_output(
    model: Any,
    tokenizer: Any,
    blocks: Sequence[torch.nn.Module],
    prompt: str,
    layer_idx: int,
    max_length: int,
) -> Tuple[Mapping[str, torch.Tensor], torch.Tensor]:
    device = jl.infer_input_device(model)
    batch = jl.encode_prompt(tokenizer, prompt, max_length, device)
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        captured["hidden"] = jl.first_hidden_from_block_output(output).detach()
        return output

    handle = blocks[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(**batch, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"failed to capture layer {layer_idx} output")
    return batch, captured["hidden"]


def simple_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # All prompts in this diagnostic are short and unpadded; GPT-OSS sliding window
    # equals ordinary causal masking at these lengths.
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def rmsnorm_high(x: torch.Tensor, norm: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    w = norm.weight.to(device=x.device, dtype=dtype)
    eps = float(getattr(norm, "variance_epsilon", 1e-6))
    x = x.to(dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return w * x * torch.rsqrt(var + eps)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rot(x: torch.Tensor) -> torch.Tensor:
        first, second = torch.chunk(x, 2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

    return rot(q), rot(k)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def linear_high(x: torch.Tensor, module: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    weight = module.weight.to(device=x.device, dtype=dtype)
    bias = module.bias.to(device=x.device, dtype=dtype) if getattr(module, "bias", None) is not None else None
    return F.linear(x.to(dtype), weight, bias)


def attention_high(
    hidden_states: torch.Tensor,
    attn: torch.nn.Module,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, int(attn.head_dim))
    query_states = linear_high(hidden_states, attn.q_proj, dtype).view(hidden_shape).transpose(1, 2)
    key_states = linear_high(hidden_states, attn.k_proj, dtype).view(hidden_shape).transpose(1, 2)
    value_states = linear_high(hidden_states, attn.v_proj, dtype).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary(query_states, key_states, cos.to(dtype), sin.to(dtype))
    key_states = repeat_kv(key_states, int(attn.num_key_value_groups))
    value_states = repeat_kv(value_states, int(attn.num_key_value_groups))
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * float(attn.scaling)
    attn_weights = attn_weights + attention_mask
    sinks = attn.sinks.to(device=hidden_states.device, dtype=dtype).reshape(1, -1, 1, 1).expand(
        query_states.shape[0], -1, query_states.shape[-2], -1
    )
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=dtype)
    scores = probs[..., :-1]
    attn_output = torch.matmul(scores.to(dtype), value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return linear_high(attn_output, attn.o_proj, dtype)


def mlp_high(
    hidden_states: torch.Tensor,
    mlp: torch.nn.Module,
    dtype: torch.dtype,
    route_dump: Optional[Dict[str, Any]] = None,
    layer_idx: Optional[int] = None,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim).to(dtype)
    router = mlp.router
    logits = F.linear(
        flat,
        router.weight.to(device=flat.device, dtype=dtype),
        router.bias.to(device=flat.device, dtype=dtype),
    )
    top_values, indices = torch.topk(logits, int(router.top_k), dim=-1)
    scores = F.softmax(top_values, dim=1, dtype=dtype)
    if route_dump is not None and layer_idx is not None:
        route_dump[str(layer_idx)] = {
            "indices_all_positions": indices.detach().cpu().tolist(),
            "gates_all_positions": [[float(v) for v in row] for row in scores.detach().float().cpu().tolist()],
            "indices_last_position": indices.detach().cpu().tolist()[-1],
            "gates_last_position": [float(v) for v in scores.detach().float().cpu().tolist()[-1]],
        }

    experts = mlp.experts
    next_states = torch.zeros_like(flat, dtype=dtype, device=flat.device)
    with torch.no_grad():
        expert_mask = F.one_hot(indices, num_classes=int(experts.num_experts)).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = flat[token_idx]
        gate_up = (
            current_state @ experts.gate_up_proj[expert_idx].to(device=flat.device, dtype=dtype)
            + experts.gate_up_proj_bias[expert_idx].to(device=flat.device, dtype=dtype)
        )
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=float(experts.limit))
        up = up.clamp(min=-float(experts.limit), max=float(experts.limit))
        glu = gate * torch.sigmoid(gate * float(experts.alpha))
        gated_output = (up + 1) * glu
        out = (
            gated_output @ experts.down_proj[expert_idx].to(device=flat.device, dtype=dtype)
            + experts.down_proj_bias[expert_idx].to(device=flat.device, dtype=dtype)
        )
        weighted = out * scores[token_idx, top_k_pos, None]
        next_states = next_states.index_add(0, token_idx, weighted)
    return next_states.reshape(batch_size, sequence_length, hidden_dim)


def downstream_high(
    model: Any,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    hidden_after_layer: torch.Tensor,
    start_layer: int,
    dtype: torch.dtype,
    route_dump: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    device = hidden_after_layer.device
    seq_len = int(input_ids.shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    # Compute RoPE at lifted dtype. This intentionally avoids bf16-only trig output.
    pos_hidden = hidden_after_layer.to(dtype)
    cos, sin = model.model.rotary_emb(pos_hidden, position_ids)
    cos = cos.to(dtype)
    sin = sin.to(dtype)
    mask = simple_causal_mask(seq_len, device, dtype)

    h = hidden_after_layer.to(dtype)
    for layer_idx in range(start_layer, len(blocks)):
        layer = blocks[layer_idx]
        residual = h
        x = rmsnorm_high(h, layer.input_layernorm, dtype)
        x = attention_high(x, layer.self_attn, (cos, sin), mask, dtype)
        h = residual + x
        residual = h
        x = rmsnorm_high(h, layer.post_attention_layernorm, dtype)
        x = mlp_high(x, layer.mlp, dtype, route_dump=route_dump, layer_idx=layer_idx)
        h = residual + x
    return h


def score_rows_high(
    model: Any,
    vector: torch.Tensor,
    token_ids: Sequence[int],
    g_weight: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Dict[int, float]:
    vec = vector.to(dtype)
    if g_weight is not None:
        vec = vec * g_weight.to(device=vec.device, dtype=dtype)
    W = model.get_output_embeddings().weight
    out: Dict[int, float] = {}
    for tid in token_ids:
        row = W[int(tid)].to(device=vec.device, dtype=dtype)
        out[int(tid)] = float((row * vec).sum().detach().float().cpu().item())
    return out


def rank_two(scores: Mapping[int, float], tid_a: int, tid_b: int) -> Dict[str, Any]:
    return {
        "orbit_minus_plant": float(scores[tid_a] - scores[tid_b]),
        "orbit_score": float(scores[tid_a]),
        "plant_score": float(scores[tid_b]),
    }


def run_high_precision_case(
    model: Any,
    tokenizer: Any,
    blocks: Sequence[torch.nn.Module],
    prompts: Sequence[str],
    query_prompt: str,
    layer_idx: int,
    position: int,
    max_length: int,
    max_pairs: int,
    position_mode: str,
    token_ids: Sequence[int],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    g_weight = jl.find_final_norm_weight(model)
    query_batch, query_hidden = capture_layer_output(model, tokenizer, blocks, query_prompt, layer_idx, max_length)
    qpos = jl.normalize_position(position, query_hidden.shape[1])
    tangent = query_hidden[0, qpos].detach().to(dtype)

    summed_jv: Optional[torch.Tensor] = None
    sums_vjp: Dict[int, Optional[torch.Tensor]] = {int(t): None for t in token_ids}
    count = 0
    route_sample: Optional[Dict[str, Any]] = None

    for prompt_idx, prompt in enumerate(prompts):
        batch, hidden = capture_layer_output(model, tokenizer, blocks, prompt, layer_idx, max_length)
        pairs = resolve_positions(batch, position_mode, max_pairs)
        for pair_idx, (source_pos, target_pos) in enumerate(pairs):
            source_pos = jl.normalize_position(source_pos, hidden.shape[1])
            target_pos = jl.normalize_position(target_pos, hidden.shape[1])

            # Path A: readout-full JVP.
            leaf = hidden.detach().to(dtype).requires_grad_(True)
            route_dump: Optional[Dict[str, Any]] = {} if route_sample is None else None
            h_final = downstream_high(model, blocks, batch["input_ids"], leaf, layer_idx + 1, dtype, route_dump=route_dump)
            if route_dump is not None:
                route_sample = {"prompt_index": prompt_idx, "pair_index": pair_idx, "routes": route_dump}
            y = h_final[0, target_pos]
            u = torch.zeros_like(y, requires_grad=True)
            (jt_u,) = torch.autograd.grad(y, leaf, grad_outputs=u, retain_graph=True, create_graph=True, allow_unused=False)
            v_at_source = torch.zeros_like(jt_u)
            v_at_source[0, source_pos] = tangent.to(device=jt_u.device, dtype=dtype)
            (jv,) = torch.autograd.grad(jt_u, u, grad_outputs=v_at_source, retain_graph=False, create_graph=False, allow_unused=False)
            vec = jv.detach().flatten()
            summed_jv = vec if summed_jv is None else summed_jv + vec
            del leaf, h_final, y, u, jt_u, v_at_source, jv

            # Path B: per-token VJP vectors, same lifted downstream function.
            leaf_b = hidden.detach().to(dtype).requires_grad_(True)
            h_final_b = downstream_high(model, blocks, batch["input_ids"], leaf_b, layer_idx + 1, dtype)
            h_vec = h_final_b[0, target_pos]
            if g_weight is not None:
                h_eff = h_vec * g_weight.to(device=h_vec.device, dtype=dtype)
            else:
                h_eff = h_vec
            for i, tid in enumerate(token_ids):
                row = model.get_output_embeddings().weight[int(tid)].to(device=h_eff.device, dtype=dtype)
                scalar = (row * h_eff).sum()
                (grad,) = torch.autograd.grad(
                    scalar,
                    leaf_b,
                    retain_graph=(i != len(token_ids) - 1),
                    create_graph=False,
                    allow_unused=False,
                )
                v = grad[0, source_pos].detach().flatten()
                old = sums_vjp[int(tid)]
                sums_vjp[int(tid)] = v if old is None else old + v
            del leaf_b, h_final_b, h_vec, h_eff
            count += 1
        jl.eprint(f"[highp {dtype}] prompt={prompt_idx + 1}/{len(prompts)} count={count}")

    if summed_jv is None or count == 0:
        raise RuntimeError("no high-precision JVP samples")
    mean_jv = summed_jv / float(count)
    path_a_scores = score_rows_high(model, mean_jv, token_ids, g_weight, dtype)

    query_vec = tangent.to(dtype)
    path_b_scores: Dict[int, float] = {}
    for tid, total in sums_vjp.items():
        if total is None:
            raise RuntimeError(f"missing VJP total for token {tid}")
        mean_v = total / float(count)
        path_b_scores[int(tid)] = float((mean_v.to(dtype) * query_vec.to(mean_v.device, dtype=dtype)).sum().detach().float().cpu().item())

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "jvp_samples": int(count),
        "path_a_readout_full": path_a_scores,
        "path_b_vjp_dictionary": path_b_scores,
        "path_a_gap": rank_two(path_a_scores, int(token_ids[0]), int(token_ids[1])),
        "path_b_gap": rank_two(path_b_scores, int(token_ids[0]), int(token_ids[1])),
        "a_minus_b_by_token": {str(t): float(path_a_scores[int(t)] - path_b_scores[int(t)]) for t in token_ids},
        "route_sample_first_calibration": route_sample,
    }


def capture_existing_routes(
    model: Any,
    tokenizer: Any,
    blocks: Sequence[torch.nn.Module],
    prompts: Sequence[str],
    layer_idx: int,
    token_ids: Sequence[int],
    max_length: int,
    max_pairs: int,
    position_mode: str,
    query_activation: torch.Tensor,
) -> Dict[str, Any]:
    capture_layers = list(range(layer_idx, len(blocks)))

    def run_with_capture(kind: str) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Any]:
        records: Dict[Tuple[int, int], Dict[str, Any]] = {}
        current_forward = {"idx": -1}
        handles = []

        def make_hook(layer: int):
            def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                if layer == layer_idx:
                    current_forward["idx"] += 1
                _, scores, indices = output
                records[(current_forward["idx"], layer)] = {
                    "indices_all_positions": indices.detach().cpu().tolist(),
                    "gates_all_positions": [[float(v) for v in row] for row in scores.detach().float().cpu().tolist()],
                    "indices_last_position": indices.detach().cpu().tolist()[-1],
                    "gates_last_position": [float(v) for v in scores.detach().float().cpu().tolist()[-1]],
                }
            return hook

        for l in capture_layers:
            handles.append(blocks[l].mlp.router.register_forward_hook(make_hook(l)))
        try:
            if kind == "A":
                ret = jl.estimate_average_jvp_for_layer(
                    model, tokenizer, blocks, prompts, layer_idx, query_activation, max_length, max_pairs, position_mode
                )
            elif kind == "B":
                ret = jl.estimate_token_vectors_for_layer(
                    model, tokenizer, blocks, prompts, layer_idx, token_ids, max_length, max_pairs, position_mode
                )
            else:
                raise ValueError(kind)
        finally:
            for h in handles:
                h.remove()
        return records, ret

    a_records, _ = run_with_capture("A")
    b_records, _ = run_with_capture("B")
    mismatches: List[Dict[str, Any]] = []
    max_gate_abs_diff = 0.0
    for key in sorted(set(a_records) | set(b_records)):
        a = a_records.get(key)
        b = b_records.get(key)
        if a is None or b is None:
            mismatches.append({"key": list(key), "reason": "missing_record", "has_A": a is not None, "has_B": b is not None})
            continue
        if a["indices_all_positions"] != b["indices_all_positions"]:
            mismatches.append({"key": list(key), "reason": "indices_mismatch", "A": a, "B": b})
        # gate weights should be numerically identical; track exact max abs either way.
        flat_a = torch.tensor(a["gates_all_positions"], dtype=torch.float32)
        flat_b = torch.tensor(b["gates_all_positions"], dtype=torch.float32)
        gate_diff = float((flat_a - flat_b).abs().max().item()) if flat_a.numel() else 0.0
        max_gate_abs_diff = max(max_gate_abs_diff, gate_diff)
        if gate_diff != 0.0:
            mismatches.append({"key": list(key), "reason": "gate_mismatch", "max_abs_diff": gate_diff, "A": a, "B": b})

    # Compact per-layer report: last-position route from the first calibration forward.
    first_forward_by_layer = []
    for l in capture_layers:
        a = a_records.get((0, l))
        b = b_records.get((0, l))
        if a is not None and b is not None:
            first_forward_by_layer.append(
                {
                    "layer": int(l),
                    "A_indices_last_position": a["indices_last_position"],
                    "A_gates_last_position": a["gates_last_position"],
                    "B_indices_last_position": b["indices_last_position"],
                    "B_gates_last_position": b["gates_last_position"],
                }
            )

    return {
        "capture_layers": capture_layers,
        "forward_count_A": 1 + max([k[0] for k in a_records] or [-1]),
        "forward_count_B": 1 + max([k[0] for k in b_records] or [-1]),
        "full_route_indices_and_gates_identical": len(mismatches) == 0,
        "max_gate_abs_diff": float(max_gate_abs_diff),
        "mismatch_count": len(mismatches),
        "mismatches_first5": mismatches[:5],
        "first_calibration_last_position_by_layer": first_forward_by_layer,
        "per_output_token_note": {
            str(tid): "routing is independent of output vocab token in both implementations; B orbit/plant share one forward graph"
            for tid in token_ids
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=str(ROOT / "gpt-oss-20b"))
    parser.add_argument("--dictionary", default=str(ROOT / "readout_full_regression_dictionary_all_layers.pt"))
    parser.add_argument("--prompts-file", default=str(ROOT / "calibration_prompts.txt"))
    parser.add_argument("--prompt", default="A spider builds a")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--out", default=str(ROOT / "layer16_orbit_plant_precision_diagnostic.json"))
    parser.add_argument("--skip-route", action="store_true")
    parser.add_argument("--skip-highp", action="store_true")
    args = parser.parse_args()

    dictionary = jl.load_dictionary(args.dictionary)
    cfg = dictionary.get("config", {}) or {}
    max_length = int(cfg.get("max_length", 128))
    max_pairs = int(cfg.get("max_pairs", 1))
    position_mode = str(cfg.get("position_mode", "last"))
    max_prompts = cfg.get("max_prompts")
    prompts = jl.load_lines(args.prompts_file, jl.DEFAULT_PROMPTS)
    if max_prompts is not None:
        prompts = prompts[: int(max_prompts)]
    jl.assert_same_calibration_prompts(dictionary, prompts)

    model, tokenizer = jl.load_model_and_tokenizer(
        args.model_id,
        torch_dtype="bfloat16",
        device_map="auto",
        load_in_4bit=False,
        trust_remote_code=True,
        local_files_only=True,
        dequantize_mxfp4="auto",
    )
    blocks, block_path = jl.find_decoder_blocks(model)
    token_ids = jl.resolve_tracked_token_ids(tokenizer, TRACK_TOKENS, None)
    token_labels = {str(tid): jl.token_label(tokenizer, tid) for tid in token_ids}

    query_activation = jl.capture_layer_activation(model, tokenizer, blocks, args.prompt, args.layer, args.position, max_length)

    result: Dict[str, Any] = {
        "prompt": args.prompt,
        "layer": int(args.layer),
        "position": int(args.position),
        "model_id": args.model_id,
        "block_path": block_path,
        "token_ids": token_ids,
        "token_labels": token_labels,
        "calibration": {
            "prompts_file": args.prompts_file,
            "num_prompts": len(prompts),
            "position_mode": position_mode,
            "max_pairs": max_pairs,
            "max_length": max_length,
        },
        "lift_scope": {
            "storage": "base MXFP4-dequantized bf16 weights left unchanged; custom downstream casts selected matmul operands on the fly",
            "lifted_for_test1": [
                "decoder blocks 17-23 q/k/v/o projection matmuls",
                "decoder blocks 17-23 router linear matmuls and gate softmax dtype",
                "decoder blocks 17-23 selected expert gate_up/down matmuls and biases",
                "decoder blocks 17-23 RMSNorm arithmetic/weights",
                "final RMSNorm gain g in readout",
                "output embedding rows for orbit/plant scoring",
            ],
            "not_lifted": [
                "embedding and blocks 0-16 used only to produce the fixed layer-16 activations",
                "full-vocab W_U rows not needed for the two-token discriminative test",
                "unused experts not selected by the route for these short prompts",
            ],
            "downstream_layers": list(range(args.layer + 1, len(blocks))),
        },
    }

    if not args.skip_route:
        jl.eprint("[diagnostic] running existing bf16 route cross-check")
        result["test2_route_crosscheck"] = capture_existing_routes(
            model, tokenizer, blocks, prompts, args.layer, token_ids, max_length, max_pairs, position_mode, query_activation
        )
        Path(args.out).write_text(jdump(result), encoding="utf-8")
        if not result["test2_route_crosscheck"]["full_route_indices_and_gates_identical"]:
            result["final_decision"] = "true_bug_route_mismatch"
            Path(args.out).write_text(jdump(result), encoding="utf-8")
            print(jdump(result))
            return

    if not args.skip_highp:
        highp_results = []
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            jl.eprint(f"[diagnostic] running lifted downstream dtype={dtype}")
            torch.cuda.empty_cache()
            highp_results.append(
                run_high_precision_case(
                    model=model,
                    tokenizer=tokenizer,
                    blocks=blocks,
                    prompts=prompts,
                    query_prompt=args.prompt,
                    layer_idx=args.layer,
                    position=args.position,
                    max_length=max_length,
                    max_pairs=max_pairs,
                    position_mode=position_mode,
                    token_ids=token_ids,
                    dtype=dtype,
                )
            )
            result["test1_precision_lift"] = highp_results
            Path(args.out).write_text(jdump(result), encoding="utf-8")
        # Validity gate against the known no-op bf16 value from the previous diagnostic.
        known_noop_orbit = 1205.37060546875
        fp32_orbit = highp_results[1]["path_a_readout_full"][str(token_ids[0])] if False else highp_results[1]["path_a_readout_full"][token_ids[0]]
        fp64_orbit = highp_results[2]["path_a_readout_full"][token_ids[0]]
        validity = {
            "known_noop_bf16_orbit_score": known_noop_orbit,
            "fp32_orbit_score": float(fp32_orbit),
            "fp64_orbit_score": float(fp64_orbit),
            "fp32_differs_from_known_noop_bit_value": float(fp32_orbit) != known_noop_orbit,
            "fp64_differs_from_known_noop_bit_value": float(fp64_orbit) != known_noop_orbit,
        }
        result["test1_validity_gate"] = validity
        # Conservative classification: only benign if route passes, validity passes,
        # and A/B differences shrink close to fp64 numerical noise.
        fp32 = highp_results[1]
        fp64 = highp_results[2]
        max_abs_ab_fp64 = max(abs(v) for v in fp64["a_minus_b_by_token"].values())
        gap_delta_fp64 = abs(fp64["path_a_gap"]["orbit_minus_plant"] - fp64["path_b_gap"]["orbit_minus_plant"])
        result["test1_convergence_summary"] = {
            "max_abs_A_minus_B_fp64_tracked_tokens": float(max_abs_ab_fp64),
            "abs_A_gap_minus_B_gap_fp64": float(gap_delta_fp64),
            "fp64_A_gap": fp64["path_a_gap"]["orbit_minus_plant"],
            "fp64_B_gap": fp64["path_b_gap"]["orbit_minus_plant"],
        }
        if not (validity["fp32_differs_from_known_noop_bit_value"] and validity["fp64_differs_from_known_noop_bit_value"]):
            result["final_decision"] = "still_undetermined_invalid_reference"
        elif max_abs_ab_fp64 < 1e-3 and gap_delta_fp64 < 1e-3:
            result["final_decision"] = "benign_precision_instability"
        else:
            result["final_decision"] = "true_bug_or_still_undetermined_highp_paths_do_not_converge"

    Path(args.out).write_text(jdump(result), encoding="utf-8")
    print(jdump(result))


if __name__ == "__main__":
    main()
