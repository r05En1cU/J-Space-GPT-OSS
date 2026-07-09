#!/usr/bin/env python3
"""Mutation tests for the readout-full regression gate.

The script computes one known-good readout-full score vector for a passing layer,
then injects representative path-A bugs and requires verify_full_scores_against_dictionary
to reject each mutated vector.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

import jspace_gpt_oss as js


DEFAULT_MODEL_PATH = "/ai/mount/stlsy/workspace/J-Space-GPT-OSS/gpt-oss-20b"
DEFAULT_DICTIONARY = "/ai/mount/stlsy/workspace/J-Space-GPT-OSS/readout_full_regression_dictionary_all_layers.pt"
DEFAULT_PROMPTS = "/ai/mount/stlsy/workspace/J-Space-GPT-OSS/calibration_prompts.txt"
DEFAULT_OUT = "/ai/mount/stlsy/workspace/J-Space-GPT-OSS/readout_full_gate_mutation_test_v2.json"


def estimate_average_vjp_bug_for_layer(
    model: Any,
    tokenizer: Any,
    blocks: Sequence[torch.nn.Module],
    prompts: Sequence[str],
    layer_idx: int,
    tangent: Tensor,
    max_length: int,
    max_pairs: int,
    position_mode: str,
) -> Tuple[Tensor, int]:
    """Deliberately wrong bug: compute mean J_q^T · tangent instead of mean J_q · tangent."""
    if not prompts:
        raise ValueError("No calibration prompts provided for mutation test.")
    device = js.infer_input_device(model)
    tangent_cpu = tangent.detach().float().cpu().flatten()
    summed: Optional[Tensor] = None
    count = 0
    for prompt_idx, prompt in enumerate(prompts):
        batch = js.encode_prompt(tokenizer, prompt, max_length, device)
        seq_len = int(batch["input_ids"].shape[1])
        pairs = js.make_position_pairs(seq_len, position_mode, max_pairs)
        for source_pos, target_pos in pairs:
            outputs, leaf, h_final = js.forward_with_layer_leaf(model, blocks, layer_idx, batch)
            source_pos = js.normalize_position(source_pos, leaf.shape[1])
            target_pos = js.normalize_position(target_pos, h_final.shape[1])
            y = h_final[0, target_pos]
            v_at_target = torch.zeros_like(y)
            v_at_target[:] = tangent_cpu.to(device=y.device, dtype=y.dtype)
            (Jt_v,) = torch.autograd.grad(
                y,
                leaf,
                grad_outputs=v_at_target,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            vec = Jt_v[0, source_pos].detach().float().cpu().flatten()
            if summed is None:
                summed = torch.zeros_like(vec)
            summed.add_(vec)
            count += 1
            del outputs, leaf, h_final, y, v_at_target, Jt_v
        js.eprint(
            f"mutation transpose-J layer={layer_idx} prompt={prompt_idx + 1}/{len(prompts)} "
            f"pairs={len(pairs)} vjp_count={count}"
        )
    if summed is None or count == 0:
        raise RuntimeError("No VJP samples were estimated for mutation test.")
    return summed / float(count), count


def run_gate(scores: Tensor, activation: Tensor, dictionary: Mapping[str, Any], layer: int, tokenizer: Any) -> Dict[str, Any]:
    try:
        gate = js.verify_full_scores_against_dictionary(scores, activation, dictionary, layer, tokenizer)
        caught = False
    except js.RegressionGateError as exc:
        gate = dict(exc.result)
        caught = True
    return {
        "gate_caught_mutation": bool(caught),
        "gate_passed": bool(gate.get("passed", False)),
        "max_relative_error": float(gate.get("max_relative_error", float("nan"))),
        "numeric_pass": bool(gate.get("numeric_pass", False)),
        "top10_set_match": bool(gate.get("top10_set_match", False)),
        "topk_inversions_sub_ulp_warn": int(gate.get("topk_inversions_sub_ulp_warn", 0)),
        "topk_inversions_large_gap_fail": int(gate.get("topk_inversions_large_gap_fail", 0)),
        "spearman": float(gate.get("spearman", float("nan"))),
        "gate": gate,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run readout-full regression-gate mutation tests.")
    js.add_model_args(parser)
    parser.set_defaults(model_id=DEFAULT_MODEL_PATH, local_files_only=True)
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY)
    parser.add_argument("--prompts-file", default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt", default="A spider builds a")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--skip-layer", type=int, default=None)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--position-mode", choices=["last", "all-same", "causal-window"])
    parser.add_argument("--out-json", default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dictionary = js.load_dictionary(args.dictionary)
    cfg = dictionary.get("config", {}) or {}
    max_length = int(args.max_length if args.max_length is not None else cfg.get("max_length", 128))
    max_prompts = args.max_prompts if args.max_prompts is not None else cfg.get("max_prompts")
    max_pairs = int(args.max_pairs if args.max_pairs is not None else cfg.get("max_pairs", 1))
    position_mode = args.position_mode or str(cfg.get("position_mode", "last"))
    prompts = js.load_lines(args.prompts_file, js.DEFAULT_PROMPTS)
    if max_prompts is not None:
        prompts = prompts[: int(max_prompts)]
    calibration_meta = js.assert_same_calibration_prompts(dictionary, prompts)

    model, tokenizer = js.load_model_and_tokenizer(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dequantize_mxfp4=args.dequantize_mxfp4,
    )
    blocks, block_path = js.find_decoder_blocks(model)
    if args.skip_layer is None:
        skip_layer = min(args.layer + 1, len(blocks) - 1) if args.layer < len(blocks) - 1 else args.layer - 1
    else:
        skip_layer = int(args.skip_layer)
    if skip_layer == args.layer:
        raise ValueError("skip-layer mutation must use a different layer")

    g_weight = js.find_final_norm_weight(model)
    activation = js.capture_layer_activation(model, tokenizer, blocks, args.prompt, args.layer, args.position, max_length)
    mean_jv, jvp_count = js.estimate_average_jvp_for_layer(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        prompts=prompts,
        layer_idx=args.layer,
        tangent=activation,
        max_length=max_length,
        max_pairs=max_pairs,
        position_mode=position_mode,
    )
    base_vector = js.apply_final_norm_gain_to_vector(mean_jv, g_weight)
    base_scores = js.score_full_vocab_from_vector(model, base_vector)
    base_gate = js.verify_full_scores_against_dictionary(base_scores, activation, dictionary, args.layer, tokenizer)

    lp = js.layer_payload(dictionary, args.layer)
    token_ids = [int(x) for x in lp["token_ids"]]
    dict_vectors = lp["vectors"].float().cpu()
    dict_scores = dict_vectors @ activation.detach().float().cpu()
    candidate_rms = float(torch.sqrt(torch.mean(dict_scores.square())).item())
    if not math.isfinite(candidate_rms) or candidate_rms <= 0:
        candidate_rms = 1.0

    mutations: List[Tuple[str, Tensor]] = []
    mutations.append(("scale_x2", base_scores * 2.0))

    vjp_bug, vjp_count = estimate_average_vjp_bug_for_layer(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        prompts=prompts,
        layer_idx=args.layer,
        tangent=activation,
        max_length=max_length,
        max_pairs=max_pairs,
        position_mode=position_mode,
    )
    mutations.append(("transpose_Jt_v_instead_of_J_v", js.score_full_vocab_from_vector(model, js.apply_final_norm_gain_to_vector(vjp_bug, g_weight))))

    mutations.append(("omit_diag_g", js.score_full_vocab_from_vector(model, mean_jv)))

    wrong_jv, wrong_count = js.estimate_average_jvp_for_layer(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        prompts=prompts,
        layer_idx=skip_layer,
        tangent=activation,
        max_length=max_length,
        max_pairs=max_pairs,
        position_mode=position_mode,
    )
    mutations.append((f"wrong_layer_{skip_layer}_J", js.score_full_vocab_from_vector(model, js.apply_final_norm_gain_to_vector(wrong_jv, g_weight))))

    mutations.append(("additive_offset_plus_0p25_rms", base_scores + 0.25 * candidate_rms))
    mutations.append(("extra_softmax_then_rescale", torch.softmax(base_scores.float(), dim=0) * candidate_rms))

    mutation_rows: List[Dict[str, Any]] = []
    for name, scores in mutations:
        row = run_gate(scores, activation, dictionary, args.layer, tokenizer)
        row["mutation"] = name
        mutation_rows.append(row)

    all_caught = all(row["gate_caught_mutation"] for row in mutation_rows)
    payload = {
        "prompt": args.prompt,
        "layer": int(args.layer),
        "position": int(args.position),
        "model_id": args.model_id,
        "block_path": block_path,
        "dictionary": args.dictionary,
        "calibration": {
            "prompts_file": args.prompts_file,
            "num_prompts": len(prompts),
            "prompts_sha256": calibration_meta["prompts_sha256"],
            "position_mode": position_mode,
            "max_pairs": max_pairs,
            "max_length": max_length,
        },
        "base": {
            "jvp_samples": int(jvp_count),
            "passed": bool(base_gate["passed"]),
            "max_relative_error": float(base_gate["max_relative_error"]),
        },
        "mutation_aux_counts": {"transpose_vjp_samples": int(vjp_count), "wrong_layer_jvp_samples": int(wrong_count)},
        "mutations": mutation_rows,
        "all_mutations_caught": bool(all_caught),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out_path), "all_mutations_caught": all_caught, "mutations": len(mutation_rows)}, ensure_ascii=False, indent=2))
    if not all_caught:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
