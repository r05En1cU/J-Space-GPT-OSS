#!/usr/bin/env python3
"""Plot formal paper-probe readout-full figures."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GROUP_COLORS = {
    "flexible-generalization": "#2a78d6",
    "probe-swap": "#1baf7a",
    "lens-eval-multihop": "#eda100",
}
JLENS = "#2a78d6"
VANILLA = "#e34948"


def load_payloads(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            payloads.extend(obj if isinstance(obj, list) else [obj])
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return payloads


def short_text(text: str, max_len: int = 58) -> str:
    one = " ".join(str(text).split())
    return one if len(one) <= max_len else one[: max_len - 1].rstrip() + "…"


def apply_axes_style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.35, alpha=0.55)
    ax.grid(False, axis="x")


def series_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    probe = payload.get("paper_probe", {}) or {}
    rows: List[Dict[str, Any]] = []
    for layer_payload in payload.get("layers", []):
        tracked = layer_payload.get("tracked", [])
        if not tracked:
            continue
        row = dict(tracked[0])
        row["layer"] = int(layer_payload["layer"])
        rows.append(row)
    rows.sort(key=lambda r: r["layer"])
    return {
        "prompt": str(payload.get("prompt", "")),
        "answer_token": str(payload.get("answer_token", "")),
        "source": str(probe.get("source", "unknown")),
        "category": str(probe.get("category", "unknown")),
        "name": str(probe.get("name", "unknown")),
        "scoring_policy": str(probe.get("scoring_policy", "unknown")),
        "rows": rows,
    }


def rank_points(rows: Sequence[Dict[str, Any]], key: str = "rank") -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            pts.append((int(row["layer"]), int(value)))
    return pts


def plot_emergence(series: Sequence[Dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.2), facecolor=PAGE)
    apply_axes_style(ax)
    by_group: Dict[str, int] = defaultdict(int)
    for item in series:
        pts = rank_points(item["rows"], "rank")
        if not pts:
            continue
        xs, ys = zip(*pts)
        color = GROUP_COLORS.get(item["source"], MUTED)
        by_group[item["source"]] += 1
        ax.plot(xs, ys, color=color, marker="o", markersize=4.2, linewidth=1.45, alpha=0.50)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Layer", color=TEXT, labelpad=8)
    ax.set_ylabel("Answer token full-vocab rank (log, lower is better)", color=TEXT, labelpad=8)
    ax.set_title("Paper-probe J-Lens answer emergence", color=TEXT, loc="left", fontsize=13, fontweight="bold", pad=12)
    subtitle = f"{len(series)} paper-origin probes; multihop scored only when explicitly included"
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, color=SECONDARY, fontsize=9, va="bottom")
    handles = [
        Line2D([0], [0], color=GROUP_COLORS.get(group, MUTED), lw=2.2, marker="o", label=f"{group} (n={count})")
        for group, count in sorted(by_group.items())
    ]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=9, labelcolor=SECONDARY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def choose_comparison(series: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    best_gain = -float("inf")
    for item in series:
        j_pts = dict(rank_points(item["rows"], "rank"))
        v_pts = dict(rank_points(item["rows"], "vanilla_rank"))
        common = sorted(set(j_pts).intersection(v_pts))
        if len(common) < 2:
            continue
        gains = [math.log10(max(v_pts[l], 1)) - math.log10(max(j_pts[l], 1)) for l in common]
        gain = sum(gains) / len(gains)
        final_gain = math.log10(max(v_pts[common[-1]], 1)) - math.log10(max(j_pts[common[-1]], 1))
        score = gain + 0.25 * final_gain
        if score > best_gain:
            best_gain = score
            best = item
    if best is None:
        raise ValueError("No series has both J-Lens and vanilla ranks.")
    return best


def plot_comparison(item: Dict[str, Any], out_path: Path) -> None:
    j_pts = rank_points(item["rows"], "rank")
    v_pts = rank_points(item["rows"], "vanilla_rank")
    if not j_pts or not v_pts:
        raise ValueError("Selected comparison series lacks rank data.")
    fig, ax = plt.subplots(figsize=(8.4, 5.4), facecolor=PAGE)
    apply_axes_style(ax)
    ax.plot(*zip(*j_pts), color=JLENS, marker="o", markersize=6.5, linewidth=2.4, label="J-Lens readout-full")
    ax.plot(*zip(*v_pts), color=VANILLA, marker="s", markersize=6.0, linewidth=2.1, linestyle="--", label="Vanilla logit-lens")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Layer", color=TEXT, labelpad=8)
    ax.set_ylabel("Answer token full-vocab rank (log, lower is better)", color=TEXT, labelpad=8)
    answer = item["answer_token"].strip() or item["answer_token"]
    ax.set_title("J-Lens vs vanilla logit-lens", color=TEXT, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.text(0.0, 1.01, f"{short_text(item['prompt'])} → {answer}", transform=ax.transAxes, color=SECONDARY, fontsize=9, va="bottom")
    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=SECONDARY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot paper-probe readout-full figures.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out-dir", default="/ai/mount/stlsy/workspace/J-Space-GPT-OSS/figures")
    parser.add_argument("--emergence-out", default="paper_readout_full_emergence_v2.png")
    parser.add_argument("--comparison-out", default="paper_jlens_vs_vanilla_logit_lens_v2.png")
    parser.add_argument("--metadata-out", default="paper_readout_full_figures_v2_metadata.json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payloads = load_payloads([Path(p) for p in args.input])
    series = [series_from_payload(p) for p in payloads]
    series = [s for s in series if s["rows"]]
    if not series:
        raise ValueError("No trajectory series found.")
    out_dir = Path(args.out_dir)
    emergence = out_dir / args.emergence_out
    comparison = out_dir / args.comparison_out
    metadata_path = out_dir / args.metadata_out
    selected = choose_comparison(series)
    plot_emergence(series, emergence)
    plot_comparison(selected, comparison)
    metadata = {
        "emergence": str(emergence),
        "comparison": str(comparison),
        "series": len(series),
        "sources": dict(sorted(defaultdict(int, {k: sum(1 for s in series if s["source"] == k) for k in {s["source"] for s in series}}).items())),
        "selected_comparison": {k: selected[k] for k in ["source", "category", "name", "prompt", "answer_token", "scoring_policy"]},
        "probes": [{k: s[k] for k in ["source", "category", "name", "prompt", "answer_token", "scoring_policy"]} for s in series],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
