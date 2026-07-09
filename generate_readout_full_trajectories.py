#!/usr/bin/env python3
"""Generate readout-full JSONL trajectories in one model load.

This is a runner around jspace_gpt_oss.py's readout-full primitives. It keeps the
approved math unchanged while avoiding one model reload per probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import jspace_gpt_oss as js

DEFAULT_PROBE_ANSWERS = {
    "The capital of France is": " Paris",
    "A spider builds a web because": " web",
    "In quantum mechanics, a particle can": " wave",
    "The recipe starts by chopping onions and": " recipe",
    "When a patient has a fever, the doctor": " medicine",
    "A compiler transforms source code into": " program",
    "The ocean tide rises when": " moon",
    "To solve the equation, first isolate": " variable",
    "A memory system stores information so that": " memory",
    "Photosynthesis allows a plant to": " plant",
    "A satellite orbits Earth because": " orbit",
}


def parse_probe(text: str) -> Tuple[str, str]:
    for sep in ("|||", "=>", "\t"):
        if sep in text:
            prompt, token = text.split(sep, 1)
            return prompt.strip(), token.rstrip("\n")
    raise ValueError("--probe must be formatted as 'prompt||| answer_token'")


def default_probes(prompts: Sequence[str], max_probes: Optional[int]) -> List[Tuple[str, str]]:
    probes: List[Tuple[str, str]] = []
    for prompt in prompts:
        for prefix, token in DEFAULT_PROBE_ANSWERS.items():
            if prompt.startswith(prefix):
                probes.append((prompt, token))
                break
        if max_probes is not None and len(probes) >= max_probes:
            break
    if not probes:
        raise ValueError("No default probes matched prompts-file; pass --probe explicitly.")
    return probes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate full-vocab J-Lens trajectory JSONL.")
    js.add_model_args(parser)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--probe", action="append", help="Probe as 'prompt||| answer_token'; repeatable.")
    parser.add_argument("--max-probes", type=int, default=3)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=8, help="Calibration prompt count for mean J.")
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--position-mode", default="last", choices=["last", "all-same", "causal-window"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-vanilla", action="store_true", help="Skip vanilla logit-lens comparison columns.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    all_prompts = js.load_lines(args.prompts_file, js.DEFAULT_PROMPTS)
    calibration_prompts = all_prompts[: args.max_prompts] if args.max_prompts is not None else all_prompts
    probes = [parse_probe(x) for x in args.probe] if args.probe else default_probes(all_prompts, args.max_probes)

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
    layers = js.parse_layers(args.layers, len(blocks))
    g_weight = js.find_final_norm_weight(model)
    if g_weight is None:
        js.eprint("[j-lens] 未找到 final-norm 可学习增益 g，trajectory 退化为 raw W_U。")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for probe_idx, (prompt, answer_token) in enumerate(probes, start=1):
            tracked_token_ids = js.resolve_tracked_token_ids(tokenizer, [answer_token], None)
            js.eprint(f"[trajectory] probe={probe_idx}/{len(probes)} prompt={prompt!r} token={answer_token!r} layers={layers}")
            layer_results: List[Dict[str, Any]] = []
            for layer_idx in layers:
                activation = js.capture_layer_activation(
                    model, tokenizer, blocks, prompt, layer_idx, args.position, args.max_length
                )
                mean_jv, jvp_count = js.estimate_average_jvp_for_layer(
                    model=model,
                    tokenizer=tokenizer,
                    blocks=blocks,
                    prompts=calibration_prompts,
                    layer_idx=layer_idx,
                    tangent=activation,
                    max_length=args.max_length,
                    max_pairs=args.max_pairs,
                    position_mode=args.position_mode,
                )
                j_lens_vector = js.apply_final_norm_gain_to_vector(mean_jv, g_weight)
                full_scores = js.score_full_vocab_from_vector(model, j_lens_vector)
                result: Dict[str, Any] = {
                    "layer": int(layer_idx),
                    "jvp_samples": int(jvp_count),
                    "top": js.rows_from_full_scores(full_scores, tokenizer, args.top_k),
                    "tracked": js.tracked_rows_from_scores(full_scores, tokenizer, tracked_token_ids),
                }
                if not args.no_vanilla:
                    vanilla_vector = js.apply_final_norm_gain_to_vector(activation, g_weight)
                    vanilla_scores = js.score_full_vocab_from_vector(model, vanilla_vector)
                    vanilla_tracked = js.tracked_rows_from_scores(
                        vanilla_scores, tokenizer, tracked_token_ids, prefix="vanilla"
                    )
                    by_tid = {int(row["token_id"]): row for row in result["tracked"]}
                    for row in vanilla_tracked:
                        by_tid[int(row["token_id"])].update(
                            {k: v for k, v in row.items() if k not in {"token_id", "text"}}
                        )
                    result["tracked"] = list(by_tid.values())
                layer_results.append(result)
            payload = {
                "prompt": prompt,
                "answer_token": answer_token,
                "position": int(args.position),
                "model_id": args.model_id,
                "block_path": block_path,
                "score_mode": "dot",
                "calibration": {
                    "prompts_file": args.prompts_file,
                    "num_prompts": len(calibration_prompts),
                    "position_mode": args.position_mode,
                    "max_pairs": int(args.max_pairs),
                    "max_length": int(args.max_length),
                },
                "layers": layer_results,
            }
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
    print(json.dumps({"out_jsonl": str(out_path), "probes": len(probes), "layers": layers}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
