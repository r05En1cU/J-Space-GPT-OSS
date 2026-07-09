#!/usr/bin/env python3
"""STEP 1 diagnostic: compare readout-full bf16-style JVP vs fp32 JVP on layer 16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

import jspace_gpt_oss as jl


def score_stats(diff: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    diff = diff.detach().float().cpu().flatten()
    ref = ref.detach().float().cpu().flatten()
    rms_ref = float(torch.linalg.norm(ref).item() / max(1, ref.numel()) ** 0.5)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rms_abs": float(torch.linalg.norm(diff).item() / max(1, diff.numel()) ** 0.5),
        "relative_to_fp32_rms_max": float(diff.abs().max().item() / max(rms_ref, jl.EPS)),
        "relative_to_fp32_rms_mean": float(diff.abs().mean().item() / max(rms_ref, jl.EPS)),
        "fp32_score_rms": rms_ref,
    }


def rank(scores: torch.Tensor, token_id: int) -> int:
    s = scores.detach().float().cpu().flatten()
    tid = int(token_id)
    return int((s > s[tid]).sum().item()) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="/ai/mount/stlsy/workspace/J-Space-GPT-OSS/gpt-oss-20b")
    parser.add_argument("--dictionary", default="/ai/mount/stlsy/workspace/J-Space-GPT-OSS/readout_full_regression_dictionary_all_layers.pt")
    parser.add_argument("--prompts-file", default="/ai/mount/stlsy/workspace/J-Space-GPT-OSS/calibration_prompts.txt")
    parser.add_argument("--prompt", default="A spider builds a")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--out", default="/ai/mount/stlsy/workspace/J-Space-GPT-OSS/step1_readout_full_fp32_noise_layer16.json")
    args = parser.parse_args()

    dictionary = jl.load_dictionary(args.dictionary)
    verify_cfg = dictionary.get("config", {}) or {}
    max_length = int(verify_cfg.get("max_length", 128))
    max_pairs = int(verify_cfg.get("max_pairs", 1))
    position_mode = str(verify_cfg.get("position_mode", "last"))
    max_prompts = verify_cfg.get("max_prompts")
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
    g_weight = jl.find_final_norm_weight(model)

    activation = jl.capture_layer_activation(
        model, tokenizer, blocks, args.prompt, args.layer, args.position, max_length
    )

    mean_jv_bf16, count_bf16 = jl.estimate_average_jvp_for_layer(
        model, tokenizer, blocks, prompts, args.layer, activation, max_length, max_pairs, position_mode, jvp_dtype="auto"
    )
    scores_bf16 = jl.score_full_vocab_from_vector(
        model, jl.apply_final_norm_gain_to_vector(mean_jv_bf16, g_weight), cosine=False
    )

    mean_jv_fp32, count_fp32 = jl.estimate_average_jvp_for_layer(
        model, tokenizer, blocks, prompts, args.layer, activation, max_length, max_pairs, position_mode, jvp_dtype="fp32"
    )
    scores_fp32 = jl.score_full_vocab_from_vector(
        model, jl.apply_final_norm_gain_to_vector(mean_jv_fp32, g_weight), cosine=False
    )

    lp = jl.layer_payload(dictionary, args.layer)
    token_ids = [int(x) for x in lp["token_ids"]]
    token_index = torch.tensor(token_ids, dtype=torch.long)
    dict_vectors = lp["vectors"].float().cpu()
    dict_scores = dict_vectors @ activation.detach().float().cpu()
    slice_bf16 = scores_bf16.detach().float().cpu()[token_index]
    slice_fp32 = scores_fp32.detach().float().cpu()[token_index]

    rho_bf16 = jl.spearman_corr(slice_bf16, dict_scores)
    rho_fp32 = jl.spearman_corr(slice_fp32, dict_scores)
    n_top = min(10, len(token_ids))
    bf16_order = torch.argsort(slice_bf16, descending=True).tolist()
    fp32_order = torch.argsort(slice_fp32, descending=True).tolist()
    dict_order = torch.argsort(dict_scores, descending=True).tolist()
    bf16_top10 = [token_ids[i] for i in bf16_order[:n_top]]
    fp32_top10 = [token_ids[i] for i in fp32_order[:n_top]]
    dict_top10 = [token_ids[i] for i in dict_order[:n_top]]

    tracked_names = [" orbit", " plant"]
    tracked: List[Dict[str, Any]] = []
    for name in tracked_names:
        tid = jl.resolve_single_token_id(tokenizer, name, None)
        dict_val = None
        dict_rank = None
        if tid in token_ids:
            idx = token_ids.index(tid)
            dict_val = float(dict_scores[idx].item())
            dict_rank = int((dict_scores > dict_scores[idx]).sum().item()) + 1
        tracked.append(
            {
                "token": name,
                "token_id": int(tid),
                "bf16_score": float(scores_bf16[tid].item()),
                "fp32_score": float(scores_fp32[tid].item()),
                "abs_fp32_minus_bf16": float(abs(scores_fp32[tid].item() - scores_bf16[tid].item())),
                "bf16_full_vocab_rank": rank(scores_bf16, tid),
                "fp32_full_vocab_rank": rank(scores_fp32, tid),
                "dictionary_score": dict_val,
                "dictionary_candidate_rank": dict_rank,
            }
        )

    orbit_id = jl.resolve_single_token_id(tokenizer, " orbit", None)
    plant_id = jl.resolve_single_token_id(tokenizer, " plant", None)
    gap = {
        "bf16_orbit_minus_plant": float(scores_bf16[orbit_id].item() - scores_bf16[plant_id].item()),
        "fp32_orbit_minus_plant": float(scores_fp32[orbit_id].item() - scores_fp32[plant_id].item()),
    }
    if orbit_id in token_ids and plant_id in token_ids:
        oi = token_ids.index(orbit_id)
        pi = token_ids.index(plant_id)
        gap["dictionary_orbit_minus_plant"] = float(dict_scores[oi].item() - dict_scores[pi].item())

    full_diff = scores_fp32 - scores_bf16
    candidate_diff = slice_fp32 - slice_bf16
    payload = {
        "step": 1,
        "prompt": args.prompt,
        "layer": int(args.layer),
        "model_id": args.model_id,
        "block_path": block_path,
        "calibration": {
            "prompts_file": args.prompts_file,
            "num_prompts": len(prompts),
            "position_mode": position_mode,
            "max_pairs": max_pairs,
            "max_length": max_length,
        },
        "jvp_samples": {"bf16": int(count_bf16), "fp32": int(count_fp32)},
        "noise_floor_full_vocab": score_stats(full_diff, scores_fp32),
        "noise_floor_dictionary_candidates": score_stats(candidate_diff, slice_fp32),
        "verify_vs_dictionary": {
            "bf16": {
                "spearman": float(rho_bf16),
                "top10_set_equal": set(bf16_top10) == set(dict_top10),
                "top10_same_order": bf16_top10 == dict_top10,
                "top10_intersection_size": len(set(bf16_top10).intersection(dict_top10)),
                "top10": [{"token_id": int(t), "text": jl.token_label(tokenizer, int(t))} for t in bf16_top10],
            },
            "fp32": {
                "spearman": float(rho_fp32),
                "top10_set_equal": set(fp32_top10) == set(dict_top10),
                "top10_same_order": fp32_top10 == dict_top10,
                "top10_intersection_size": len(set(fp32_top10).intersection(dict_top10)),
                "top10": [{"token_id": int(t), "text": jl.token_label(tokenizer, int(t))} for t in fp32_top10],
            },
            "dictionary_top10": [{"token_id": int(t), "text": jl.token_label(tokenizer, int(t))} for t in dict_top10],
        },
        "tracked_tokens": tracked,
        "orbit_plant_gap": gap,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
