#!/usr/bin/env python3
"""Phase 0 read-only GPU measurements for MoE routing x J-Space.

This script imports the established model/JVP helpers but does not modify the core
implementation or regression gates. It measures:
  1) token-wise Jv cancellation for random and router-boundary directions;
  2) one-pass randomized SVD sketches of prompt-local token-averaged J_l and M_l.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import jspace_gpt_oss as jl


def now() -> float:
    return time.perf_counter()


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def finite_float(x: float) -> Optional[float]:
    return float(x) if math.isfinite(float(x)) else None


def quantiles(values: torch.Tensor) -> Dict[str, float]:
    x = values.detach().float().cpu().flatten()
    qs = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    y = torch.quantile(x, qs)
    return {k: float(v) for k, v in zip(["min", "p10", "p25", "median", "p75", "p90", "max"], y.tolist())}


def selected_positions(seq_len: int, max_positions: int) -> List[int]:
    if max_positions <= 0 or seq_len <= max_positions:
        return list(range(seq_len))
    raw = torch.linspace(0, seq_len - 1, max_positions).round().long().tolist()
    return sorted(set(int(x) for x in raw))


def batched_jvp_from_graph(
    leaf: torch.Tensor,
    h_final: torch.Tensor,
    position: int,
    directions: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Return rows [k,d] containing J_position @ direction_i."""
    y = h_final[0, position]
    u = torch.zeros_like(y, requires_grad=True)
    (jt_u,) = torch.autograd.grad(
        y,
        leaf,
        grad_outputs=u,
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )
    rows: List[torch.Tensor] = []
    for start in range(0, int(directions.shape[0]), int(batch_size)):
        v = directions[start : start + batch_size].to(device=jt_u.device, dtype=jt_u.dtype)
        grad_batch = torch.zeros(
            (int(v.shape[0]),) + tuple(jt_u.shape), device=jt_u.device, dtype=jt_u.dtype
        )
        grad_batch[:, 0, position, :] = v
        try:
            (jv,) = torch.autograd.grad(
                jt_u,
                u,
                grad_outputs=grad_batch,
                is_grads_batched=True,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            rows.append(jv.detach().float().cpu())
        except Exception as exc:
            print(f"[phase0] batched JVP fallback at position={position}: {type(exc).__name__}: {exc}", flush=True)
            fallback: List[torch.Tensor] = []
            for i in range(int(v.shape[0])):
                (one,) = torch.autograd.grad(
                    jt_u,
                    u,
                    grad_outputs=grad_batch[i],
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )
                fallback.append(one.detach().float().cpu())
            rows.append(torch.stack(fallback, dim=0))
    return torch.cat(rows, dim=0)


def batched_vjp_from_graph(
    leaf: torch.Tensor,
    h_final: torch.Tensor,
    position: int,
    cotangents: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Return rows [k,d] containing J_position^T @ cotangent_i."""
    y = h_final[0, position]
    rows: List[torch.Tensor] = []
    for start in range(0, int(cotangents.shape[0]), int(batch_size)):
        u = cotangents[start : start + batch_size].to(device=y.device, dtype=y.dtype)
        try:
            (vjp,) = torch.autograd.grad(
                y,
                leaf,
                grad_outputs=u,
                is_grads_batched=True,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            rows.append(vjp[:, 0, position, :].detach().float().cpu())
        except Exception as exc:
            print(f"[phase0] batched VJP fallback at position={position}: {type(exc).__name__}: {exc}", flush=True)
            fallback: List[torch.Tensor] = []
            for i in range(int(u.shape[0])):
                (one,) = torch.autograd.grad(
                    y,
                    leaf,
                    grad_outputs=u[i],
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )
                fallback.append(one[0, position].detach().float().cpu())
            rows.append(torch.stack(fallback, dim=0))
    return torch.cat(rows, dim=0)


def average_jvp_matrix(
    model: Any,
    blocks: Sequence[torch.nn.Module],
    layer: int,
    batch: Mapping[str, torch.Tensor],
    positions: Sequence[int],
    directions_dk: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Compute prompt-local mean_position J_position @ directions, shape [d,k]."""
    with jl.patched_transformers_moe_grouped_mm_for_double_backward():
        outputs, leaf, h_final = jl.forward_with_layer_leaf(model, blocks, layer, batch)
        acc = torch.zeros((int(directions_dk.shape[1]), int(directions_dk.shape[0])), dtype=torch.float32)
        directions = directions_dk.T.contiguous()
        for index, pos in enumerate(positions, start=1):
            jv = batched_jvp_from_graph(leaf, h_final, int(pos), directions, batch_size)
            acc.add_(jv)
            print(f"[phase0] layer={layer} JVP position={index}/{len(positions)} k={directions.shape[0]}", flush=True)
        del outputs, leaf, h_final
    return (acc / float(len(positions))).T.contiguous()


def average_vjp_matrix(
    model: Any,
    blocks: Sequence[torch.nn.Module],
    layer: int,
    batch: Mapping[str, torch.Tensor],
    positions: Sequence[int],
    cotangents_dk: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Compute prompt-local mean_position J_position^T @ cotangents, shape [d,k]."""
    with jl.patched_transformers_moe_grouped_mm_for_double_backward():
        outputs, leaf, h_final = jl.forward_with_layer_leaf(model, blocks, layer, batch)
        acc = torch.zeros_like(cotangents_dk, dtype=torch.float32, device="cpu")
        cotangents = cotangents_dk.T.contiguous()
        for index, pos in enumerate(positions, start=1):
            vjp = batched_vjp_from_graph(leaf, h_final, int(pos), cotangents, batch_size)
            acc.add_(vjp.T)
            print(f"[phase0] layer={layer} VJP position={index}/{len(positions)} k={cotangents.shape[0]}", flush=True)
        del outputs, leaf, h_final
    return acc / float(len(positions))


def capture_router_competition(
    model: Any,
    blocks: Sequence[torch.nn.Module],
    batch: Mapping[str, torch.Tensor],
    layer: int,
    tokenizer: Any,
    n_directions: int,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    captured: Dict[str, Any] = {}

    def hook(_module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        raw, gates, indices = output
        captured["router_input"] = inputs[0].detach().float().cpu()
        captured["raw"] = raw.detach().float().cpu() if torch.is_tensor(raw) else None
        captured["gates"] = gates.detach().float().cpu()
        captured["indices"] = indices.detach().cpu()

    handle = blocks[layer].mlp.router.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**batch, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    gates = captured["gates"]
    indices = captured["indices"]
    if gates.dim() == 3:
        gates = gates.reshape(-1, gates.shape[-1])
        indices = indices.reshape(-1, indices.shape[-1])
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    candidates: List[Tuple[float, int, int, int]] = []
    for pos in range(int(indices.shape[0])):
        i = int(indices[pos, 0].item())
        j = int(indices[pos, 1].item())
        gap = abs(float(gates[pos, 0].item()) - float(gates[pos, 1].item()))
        candidates.append((gap, pos, i, j))
    candidates.sort(key=lambda x: x[0])
    router = blocks[layer].mlp.router
    gamma = blocks[layer].post_attention_layernorm.weight.detach().float().cpu()
    weight = router.weight.detach().float().cpu()
    records: List[Dict[str, Any]] = []
    directions: List[torch.Tensor] = []
    seen = set()
    for gap, pos, i, j in candidates:
        pair = tuple(sorted((i, j)))
        if pair in seen:
            continue
        seen.add(pair)
        delta_raw = gamma * (weight[i] - weight[j])
        delta = F.normalize(delta_raw, dim=0)
        directions.append(delta)
        token_id = int(input_ids[pos]) if pos < len(input_ids) else -1
        records.append(
            {
                "position": int(pos),
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]) if token_id >= 0 else None,
                "expert_i": i,
                "expert_j": j,
                "top_gate": float(gates[pos, 0].item()),
                "second_gate": float(gates[pos, 1].item()),
                "gate_gap": float(gap),
                "delta_raw_norm": float(delta_raw.norm().item()),
            }
        )
        if len(directions) >= int(n_directions):
            break
    if len(directions) < int(n_directions):
        raise RuntimeError(f"Only found {len(directions)} unique router expert pairs")
    return records, torch.stack(directions, dim=0)


def response_metrics(responses_pkd: torch.Tensor) -> Dict[str, Any]:
    """Summarize token-wise responses for each direction."""
    p, k, _ = responses_pkd.shape
    rows: List[Dict[str, Any]] = []
    for direction in range(k):
        r = responses_pkd[:, direction, :].float()
        norms = torch.linalg.vector_norm(r, dim=1)
        mean = r.mean(dim=0)
        mean_norm = float(mean.norm().item())
        mean_amp = float(norms.mean().item())
        ratio = mean_norm / max(mean_amp, 1e-12)
        rn = F.normalize(r, dim=1)
        pair_cos = rn @ rn.T
        tri = pair_cos[torch.triu_indices(p, p, offset=1).unbind()]
        if mean_norm > 1e-12:
            projections = r @ (mean / mean.norm())
            sign_fraction = float((projections >= 0).float().mean().item())
        else:
            projections = torch.zeros(p)
            sign_fraction = float("nan")
        rows.append(
            {
                "mean_response_norm": mean_norm,
                "per_token_norm": quantiles(norms),
                "mean_per_token_norm": mean_amp,
                "mean_over_token_amplitude_ratio": float(ratio),
                "projection_on_mean_sign_fraction": finite_float(sign_fraction),
                "projection_on_mean": [float(x) for x in projections.tolist()],
                "pairwise_cosine_mean": float(tri.mean().item()) if tri.numel() else 1.0,
                "pairwise_nonnegative_cosine_fraction": float((tri >= 0).float().mean().item()) if tri.numel() else 1.0,
            }
        )
    return {"num_positions": int(p), "directions": rows}


def run_mismatch(args: argparse.Namespace, model: Any, tokenizer: Any, blocks: Sequence[torch.nn.Module]) -> Dict[str, Any]:
    device = jl.infer_input_device(model)
    batch = jl.encode_prompt(tokenizer, args.prompt, args.max_length, device)
    seq_len = int(batch["input_ids"].shape[1])
    positions = selected_positions(seq_len, args.max_positions)
    router_records, router_dirs = capture_router_competition(
        model, blocks, batch, args.layer, tokenizer, args.router_directions
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    random_dirs = unit_rows(torch.randn(args.random_directions, router_dirs.shape[1], generator=generator))
    directions = torch.cat([random_dirs, router_dirs], dim=0)
    labels = [f"random_{i}" for i in range(args.random_directions)] + [
        f"router_e{r['expert_i']}_vs_e{r['expert_j']}_pos{r['position']}" for r in router_records
    ]
    with jl.patched_transformers_moe_grouped_mm_for_double_backward():
        outputs, leaf, h_final = jl.forward_with_layer_leaf(model, blocks, args.layer, batch)
        responses: List[torch.Tensor] = []
        for index, pos in enumerate(positions, start=1):
            responses.append(batched_jvp_from_graph(leaf, h_final, pos, directions, args.jvp_batch_size))
            print(f"[phase0] mismatch layer={args.layer} position={index}/{len(positions)}", flush=True)
        del outputs, leaf, h_final
    tensor = torch.stack(responses, dim=0)
    metrics = response_metrics(tensor)
    for label, kind, row in zip(
        labels,
        ["random"] * args.random_directions + ["router_boundary"] * args.router_directions,
        metrics["directions"],
    ):
        row["label"] = label
        row["kind"] = kind
    return {
        "measurement": "average_J_vs_per_token_J",
        "layer": int(args.layer),
        "prompt": args.prompt,
        "input_tokens": [tokenizer.decode([int(x)]) for x in batch["input_ids"][0].detach().cpu().tolist()],
        "positions": positions,
        "router_direction_definition": "normalize(gamma * (router_weight_i - router_weight_j)); gamma=post_attention_layernorm.weight",
        "router_competitions": router_records,
        "metrics": metrics,
    }


def apply_w_eff(
    model: Any,
    g_cpu: Optional[torch.Tensor],
    x_dk: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    weight = model.get_output_embeddings().weight.detach()
    device = weight.device
    x = x_dk.to(device=device, dtype=torch.float32)
    g = None if g_cpu is None else g_cpu.to(device=device, dtype=torch.float32)
    chunks: List[torch.Tensor] = []
    for start in range(0, int(weight.shape[0]), int(chunk_size)):
        end = min(int(weight.shape[0]), start + int(chunk_size))
        w = weight[start:end].to(dtype=torch.float32)
        if g is not None:
            w = w * g.unsqueeze(0)
        chunks.append((w @ x).detach().cpu())
    return torch.cat(chunks, dim=0)


def apply_w_eff_t(
    model: Any,
    g_cpu: Optional[torch.Tensor],
    q_vk: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    weight = model.get_output_embeddings().weight.detach()
    device = weight.device
    g = None if g_cpu is None else g_cpu.to(device=device, dtype=torch.float32)
    acc = torch.zeros((int(weight.shape[1]), int(q_vk.shape[1])), device=device, dtype=torch.float32)
    for start in range(0, int(weight.shape[0]), int(chunk_size)):
        end = min(int(weight.shape[0]), start + int(chunk_size))
        w = weight[start:end].to(dtype=torch.float32)
        if g is not None:
            w = w * g.unsqueeze(0)
        q = q_vk[start:end].to(device=device, dtype=torch.float32)
        acc.add_(w.T @ q)
    return acc.detach().cpu()


def spectrum_summary(
    singular_top: torch.Tensor,
    frob_sq_est: float,
    d: int,
    threshold_rel: float,
    restricted_singular: torch.Tensor,
) -> Dict[str, Any]:
    s = singular_top.detach().float().cpu()
    sigma_max = float(s.max().item())
    threshold = float(threshold_rel * sigma_max)
    stable_rank = float(frob_sq_est / max(sigma_max * sigma_max, 1e-30))
    denom = max(sigma_max * sigma_max - threshold * threshold, 1e-30)
    trace_count_est = (frob_sq_est - d * threshold * threshold) / denom
    trace_count_est = max(0.0, min(float(d), float(trace_count_est)))
    restricted = restricted_singular.detach().float().cpu()
    return {
        "sigma_max_est": sigma_max,
        "sigma_min_global": None,
        "sigma_min_global_note": "Not identified by randomized range SVD; restricted-subspace minimum and identity-residual bound are reported separately.",
        "sigma_k_est": float(s.min().item()),
        "top_singular_values": [float(x) for x in s[: min(12, s.numel())].tolist()],
        "frob_norm_est": float(math.sqrt(max(frob_sq_est, 0.0))),
        "rms_singular_est": float(math.sqrt(max(frob_sq_est / d, 0.0))),
        "stable_rank_est": stable_rank,
        "effective_rank_threshold": threshold,
        "effective_rank_trace_based_est": trace_count_est,
        "restricted_subspace_sigma": quantiles(restricted),
        "restricted_rank_above_threshold": int((restricted > threshold).sum().item()),
        "restricted_dimension": int(restricted.numel()),
    }


def run_spectrum_layer(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    blocks: Sequence[torch.nn.Module],
    layer: int,
) -> Dict[str, Any]:
    start_time = now()
    device = jl.infer_input_device(model)
    batch = jl.encode_prompt(tokenizer, args.prompt, args.max_length, device)
    seq_len = int(batch["input_ids"].shape[1])
    positions = selected_positions(seq_len, args.max_positions)
    d = int(getattr(model.config, "hidden_size", 0) or model.get_input_embeddings().weight.shape[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1000 * int(layer))
    omega = torch.randn(d, args.probes, generator=generator, dtype=torch.float32)
    q_in = torch.linalg.qr(omega, mode="reduced").Q.contiguous()

    t_jq = average_jvp_matrix(model, blocks, layer, batch, positions, q_in, args.jvp_batch_size)
    q_out_j = torch.linalg.qr(t_jq, mode="reduced").Q.contiguous()
    a_q = t_jq - q_in
    q_out_a = torch.linalg.qr(a_q, mode="reduced").Q.contiguous()

    g = jl.find_final_norm_weight(model)
    g_cpu = None if g is None else g.detach().float().cpu()
    y_m = apply_w_eff(model, g_cpu, t_jq, args.w_chunk_size)
    q_out_m = torch.linalg.qr(y_m, mode="reduced").Q.contiguous()
    w_t_qm = apply_w_eff_t(model, g_cpu, q_out_m, args.w_chunk_size)

    all_cotangents = torch.cat([q_out_j, q_out_a, w_t_qm], dim=1)
    all_vjp = average_vjp_matrix(
        model, blocks, layer, batch, positions, all_cotangents, args.vjp_batch_size
    )
    k = int(args.probes)
    z_j = all_vjp[:, :k]
    z_a = all_vjp[:, k : 2 * k] - q_out_a
    z_m = all_vjp[:, 2 * k :]

    b_j = z_j.T.contiguous()
    b_a = z_a.T.contiguous()
    b_m = z_m.T.contiguous()
    s_j = torch.linalg.svdvals(b_j)
    s_a = torch.linalg.svdvals(b_a)
    s_m = torch.linalg.svdvals(b_m)
    restricted_j = torch.linalg.svdvals(t_jq)
    restricted_m = torch.linalg.svdvals(y_m)

    scale = float(d) / float(args.probes)
    frob_j_sq = scale * float(t_jq.square().sum().item())
    frob_a_sq = scale * float(a_q.square().sum().item())
    frob_m_sq = scale * float(y_m.square().sum().item())
    j_summary = spectrum_summary(s_j, frob_j_sq, d, args.rank_threshold_rel, restricted_j)
    a_sigma_max = float(s_a.max().item())
    j_summary.update(
        {
            "J_minus_I_sigma_max_est": a_sigma_max,
            "J_minus_I_frob_norm_est": float(math.sqrt(max(frob_a_sq, 0.0))),
            "identity_relative_frob": float(math.sqrt(max(frob_a_sq, 0.0)) / max(math.sqrt(max(frob_j_sq, 0.0)), 1e-30)),
            "sigma_min_identity_bound_est": float(max(0.0, 1.0 - a_sigma_max)),
            "sigma_max_identity_bound_est": float(1.0 + a_sigma_max),
            "identity_bound_note": "Uses randomized estimate of ||J-I||_2, so it is diagnostic rather than a certified bound.",
        }
    )
    m_summary = spectrum_summary(s_m, frob_m_sq, d, args.rank_threshold_rel, restricted_m)

    del y_m, q_out_m, all_cotangents, all_vjp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "layer": int(layer),
        "prompt": args.prompt,
        "tokens": [tokenizer.decode([int(x)]) for x in batch["input_ids"][0].detach().cpu().tolist()],
        "positions": positions,
        "averaging": "mean of same-position Jacobians over selected token positions in one calibration prompt",
        "method": {
            "type": "one-pass randomized SVD with Gaussian range finder",
            "probes": int(args.probes),
            "input_probe_basis": "QR-orthonormalized Gaussian",
            "J_top_spectrum": "SVD(Q_out^T J) using JQ range and a second batched VJP pass",
            "M_top_spectrum": "SVD(Q_vocab^T W_U diag(g) J) using M Q range and the same VJP pass",
            "global_sigma_min": "not identifiable from a top-range sketch; random-subspace floor and J-I diagnostic bound reported",
        },
        "J": j_summary,
        "M": m_summary,
        "elapsed_seconds": float(now() - start_time),
    }


def load_model(args: argparse.Namespace) -> Tuple[Any, Any, Sequence[torch.nn.Module], str]:
    model, tokenizer = jl.load_model_and_tokenizer(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=False,
        trust_remote_code=True,
        local_files_only=True,
        dequantize_mxfp4="auto",
    )
    blocks, path = jl.find_decoder_blocks(model)
    return model, tokenizer, blocks, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mismatch", "spectrum", "all"], default="all")
    parser.add_argument("--model-id", default="./gpt-oss-20b")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prompt", default="The legal contract states that the buyer")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--layers", default="4,16")
    parser.add_argument("--random-directions", type=int, default=3)
    parser.add_argument("--router-directions", type=int, default=3)
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--jvp-batch-size", type=int, default=2)
    parser.add_argument("--vjp-batch-size", type=int, default=8)
    parser.add_argument("--w-chunk-size", type=int, default=4096)
    parser.add_argument("--rank-threshold-rel", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    load_start = now()
    model, tokenizer, blocks, block_path = load_model(args)
    result: Dict[str, Any] = {
        "model_id": args.model_id,
        "block_path": block_path,
        "num_layers": len(blocks),
        "hidden_size": int(getattr(model.config, "hidden_size", 0) or model.get_input_embeddings().weight.shape[1]),
        "load_seconds": float(now() - load_start),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if args.mode in {"mismatch", "all"}:
        result["mismatch"] = run_mismatch(args, model, tokenizer, blocks)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode in {"spectrum", "all"}:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
        result["spectrum"] = []
        for layer in layers:
            result["spectrum"].append(run_spectrum_layer(args, model, tokenizer, blocks, layer))
            Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
