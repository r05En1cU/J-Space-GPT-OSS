#!/usr/bin/env python3
"""Generate readout-full trajectories for the paper-origin probes.

Defaults intentionally exclude lens-eval-multihop: for that dataset the target is
used to locate the readout position while intermediates are the scored J-Lens
concepts. This runner therefore uses flexible-generalization + probe-swap by
default, both of which have direct prompt -> answer-token scoring semantics.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

import jspace_gpt_oss as js

ROOT = Path("/ai/mount/stlsy/workspace/J-Space-GPT-OSS")
DEFAULT_MODEL_PATH = str(ROOT / "gpt-oss-20b")
DEFAULT_PROBE_ROOT = ROOT / "_paper_fetch" / "jacobian-lens" / "data"
DEFAULT_OUT = str(ROOT / "paper_readout_full_trajectories_v2.jsonl")
DEFAULT_LAYERS = "0,4,8,12,16,20,23"
DEFAULT_MAX_PROBES = int(os.environ.get("MAX_PROBES", "20"))


def answer_token(answer: str) -> str:
    answer = str(answer).strip()
    if not answer:
        raise ValueError("empty answer token")
    return answer if answer.startswith(" ") else " " + answer


def load_flexible_generalization(root: Path) -> List[Dict[str, Any]]:
    data = json.loads((root / "experiments" / "flexible-generalization.json").read_text(encoding="utf-8"))
    probes: List[Dict[str, Any]] = []
    for category in data["categories"]:
        cat_name = str(category["name"])
        for func in category["funcs"]:
            func_name = str(func["name"])
            template = str(func["template"])
            answers = func["answers"]
            for arg in category["args"]:
                prompt = template.format(arg=arg)
                probes.append(
                    {
                        "source": "flexible-generalization",
                        "category": cat_name,
                        "name": f"{cat_name}/{func_name}/{arg}",
                        "prompt": prompt,
                        "answer": str(answers[arg]),
                        "answer_token": answer_token(str(answers[arg])),
                        "scored_concept": str(answers[arg]),
                        "scoring_policy": "prompt_to_answer",
                    }
                )
    return probes


def load_probe_swap(root: Path) -> List[Dict[str, Any]]:
    data = json.loads((root / "experiments" / "probe-swap.json").read_text(encoding="utf-8"))
    probes: List[Dict[str, Any]] = []
    for item in data["items"]:
        answer = str(item["answer"])
        probes.append(
            {
                "source": "probe-swap",
                "category": str(item.get("category", "uncategorized")),
                "name": str(item.get("name", item["prompt"][:40])),
                "prompt": str(item["prompt"]),
                "answer": answer,
                "answer_token": answer_token(answer),
                "intermediate": item.get("intermediate"),
                "swap_to": item.get("swap_to"),
                "scored_concept": answer,
                "scoring_policy": "prompt_to_answer",
            }
        )
    return probes


def load_multihop_intermediates(root: Path) -> List[Dict[str, Any]]:
    data = json.loads((root / "evaluations" / "lens-eval-multihop.json").read_text(encoding="utf-8"))
    probes: List[Dict[str, Any]] = []
    for item in data["items"]:
        intermediates = [str(x) for x in item.get("intermediates", []) if str(x).strip()]
        if not intermediates:
            continue
        concept = intermediates[0]
        probes.append(
            {
                "source": "lens-eval-multihop",
                "category": "multihop-intermediate",
                "name": str(item.get("name", item["prompt"][:40])),
                "prompt": str(item["prompt"]),
                "answer": concept,
                "answer_token": answer_token(concept),
                "target": item.get("target"),
                "intermediates": intermediates,
                "scored_concept": concept,
                "scoring_policy": "multihop_intermediate_not_target",
            }
        )
    return probes


def balanced_subset(probes: Sequence[Dict[str, Any]], max_probes: Optional[int]) -> List[Dict[str, Any]]:
    if max_probes is None or max_probes <= 0 or len(probes) <= max_probes:
        return list(probes)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for probe in probes:
        groups[f"{probe['source']}::{probe['category']}"].append(probe)
    selected: List[Dict[str, Any]] = []
    group_keys = sorted(groups)
    cursor = 0
    while len(selected) < max_probes and any(groups.values()):
        key = group_keys[cursor % len(group_keys)]
        if groups[key]:
            selected.append(groups[key].pop(0))
        cursor += 1
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper-probe readout-full trajectory JSONL.")
    js.add_model_args(parser)
    parser.set_defaults(model_id=DEFAULT_MODEL_PATH, local_files_only=True)
    parser.add_argument("--probe-root", default=str(DEFAULT_PROBE_ROOT))
    parser.add_argument("--prompts-file", default=str(ROOT / "calibration_prompts.txt"))
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT)
    parser.add_argument("--max-probes", type=int, default=DEFAULT_MAX_PROBES)
    parser.add_argument("--include-multihop", action="store_true", help="Also score multihop intermediates, not targets.")
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=4, help="Calibration prompt count for the figure run; gate uses the full dictionary setting separately.")
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--position-mode", default="last", choices=["last", "all-same", "causal-window"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-vanilla", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    probe_root = Path(args.probe_root)
    probes = load_flexible_generalization(probe_root) + load_probe_swap(probe_root)
    if args.include_multihop:
        probes += load_multihop_intermediates(probe_root)
    selected = balanced_subset(probes, args.max_probes)
    if not selected:
        raise ValueError("No paper probes selected.")

    all_prompts = js.load_lines(args.prompts_file, js.DEFAULT_PROMPTS)
    calibration_prompts = all_prompts[: args.max_prompts] if args.max_prompts is not None else all_prompts
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
        for probe_idx, probe in enumerate(selected, start=1):
            prompt = str(probe["prompt"])
            answer = str(probe["answer_token"])
            tracked_token_ids = js.resolve_tracked_token_ids(tokenizer, [answer], None)
            js.eprint(
                f"[paper-trajectory] probe={probe_idx}/{len(selected)} source={probe['source']} "
                f"category={probe['category']} name={probe['name']} token={answer!r} layers={layers}"
            )
            layer_results: List[Dict[str, Any]] = []
            for layer_idx in layers:
                activation = js.capture_layer_activation(model, tokenizer, blocks, prompt, layer_idx, args.position, args.max_length)
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
                    vanilla_tracked = js.tracked_rows_from_scores(vanilla_scores, tokenizer, tracked_token_ids, prefix="vanilla")
                    by_tid = {int(row["token_id"]): row for row in result["tracked"]}
                    for row in vanilla_tracked:
                        by_tid[int(row["token_id"])].update({k: v for k, v in row.items() if k not in {"token_id", "text"}})
                    result["tracked"] = list(by_tid.values())
                layer_results.append(result)
            payload = {
                "prompt": prompt,
                "answer_token": answer,
                "position": int(args.position),
                "model_id": args.model_id,
                "block_path": block_path,
                "score_mode": "dot",
                "paper_probe": probe,
                "calibration": {
                    "prompts_file": args.prompts_file,
                    "num_prompts": len(calibration_prompts),
                    "prompts_sha256": js.calibration_prompts_sha256(calibration_prompts),
                    "position_mode": args.position_mode,
                    "max_pairs": int(args.max_pairs),
                    "max_length": int(args.max_length),
                },
                "layers": layer_results,
            }
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
    print(json.dumps({"out_jsonl": str(out_path), "probes": len(selected), "layers": layers}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
