#!/usr/bin/env python3
"""Plot full-vocab J-Lens readout trajectories from readout-full JSON/JSONL outputs.

Inputs are JSON payloads emitted by:
  python jspace_gpt_oss.py readout-full --layers ... --track-token ... --include-vanilla

Two figures are produced:
  1. emergence trajectory: layer vs answer-token full-vocab rank (log y), multi-probe overlay
  2. comparison: J-Lens vs vanilla logit-lens rank curve for one tracked probe
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
TEXT = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def load_payloads(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Prefer whole-file JSON for pretty JSON arrays/objects, but fall back to
        # JSONL when multiple compact payloads start with "{" on separate lines.
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            if isinstance(obj, list):
                payloads.extend(obj)
            else:
                payloads.append(obj)
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL line") from exc
    return payloads


def short_prompt(prompt: str, max_len: int = 42) -> str:
    one_line = " ".join(prompt.split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 1].rstrip() + "…"


def collect_series(payloads: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int, str], List[Dict[str, Any]]]:
    series: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        prompt = str(payload.get("prompt", ""))
        for layer_payload in payload.get("layers", []):
            layer = int(layer_payload["layer"])
            for row in layer_payload.get("tracked", []):
                token_id = int(row["token_id"])
                text = str(row.get("text", token_id))
                item = {
                    "layer": layer,
                    "rank": row.get("rank"),
                    "score": row.get("score"),
                    "vanilla_rank": row.get("vanilla_rank"),
                    "vanilla_score": row.get("vanilla_score"),
                }
                series[(prompt, token_id, text)].append(item)
    for values in series.values():
        values.sort(key=lambda x: x["layer"])
    return dict(series)


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


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_emergence(series: Dict[Tuple[str, int, str], List[Dict[str, Any]]], out_path: Path, title: str) -> None:
    if not series:
        raise ValueError("No tracked token series found. Run readout-full with --track-token/--track-token-id.")
    fig, ax = plt.subplots(figsize=(9.4, 5.6), facecolor=PAGE)
    apply_axes_style(ax)
    for idx, ((prompt, _token_id, token_text), rows) in enumerate(series.items()):
        points = [(r["layer"], r["rank"]) for r in rows if r.get("rank") is not None]
        if not points:
            continue
        xs, ys = zip(*points)
        label = f"{short_prompt(prompt)} → {token_text.strip() or token_text}"
        color = SERIES[idx] if idx < len(SERIES) else MUTED
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot(xs, ys, color=color, marker=marker, markersize=5.5, linewidth=2.0, label=label)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Layer", color=TEXT, labelpad=8)
    ax.set_ylabel("Answer token full-vocab rank (log, lower is better)", color=TEXT, labelpad=8)
    ax.set_title(title, color=TEXT, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5, labelcolor=SECONDARY)
    save_figure(fig, out_path)


def choose_comparison_series(series: Dict[Tuple[str, int, str], List[Dict[str, Any]]]) -> Tuple[Tuple[str, int, str], List[Dict[str, Any]]]:
    for key, rows in series.items():
        if any(row.get("vanilla_rank") is not None for row in rows):
            return key, rows
    raise ValueError("No vanilla_rank values found. Run readout-full with --include-vanilla.")


def plot_comparison(series: Dict[Tuple[str, int, str], List[Dict[str, Any]]], out_path: Path, title: str) -> None:
    (prompt, _token_id, token_text), rows = choose_comparison_series(series)
    j_points = [(r["layer"], r["rank"]) for r in rows if r.get("rank") is not None]
    v_points = [(r["layer"], r["vanilla_rank"]) for r in rows if r.get("vanilla_rank") is not None]
    if not j_points or not v_points:
        raise ValueError("Selected comparison series lacks J-Lens or vanilla ranks.")
    fig, ax = plt.subplots(figsize=(7.8, 5.2), facecolor=PAGE)
    apply_axes_style(ax)
    ax.plot(*zip(*j_points), color=SERIES[0], marker="o", markersize=6, linewidth=2.2, label="J-Lens (mean JVP)")
    ax.plot(*zip(*v_points), color=SERIES[5], marker="s", markersize=5.5, linewidth=2.0, linestyle="--", label="Vanilla logit-lens")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Layer", color=TEXT, labelpad=8)
    ax.set_ylabel("Answer token full-vocab rank (log, lower is better)", color=TEXT, labelpad=8)
    subtitle = f"{short_prompt(prompt)} → {token_text.strip() or token_text}"
    ax.set_title(title, color=TEXT, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, color=SECONDARY, fontsize=9, va="bottom")
    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=SECONDARY)
    save_figure(fig, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot readout-full full-vocab rank trajectories.")
    parser.add_argument("--input", action="append", required=True, help="readout-full JSON or JSONL path; repeatable.")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--emergence-out", default="readout_full_emergence.png")
    parser.add_argument("--comparison-out", default="jlens_vs_vanilla_logit_lens.png")
    parser.add_argument("--title", default="Full-vocab J-Lens emergence trajectories")
    parser.add_argument("--comparison-title", default="J-Lens vs vanilla logit-lens")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    payloads = load_payloads([Path(p) for p in args.input])
    series = collect_series(payloads)
    emergence = out_dir / args.emergence_out
    comparison = out_dir / args.comparison_out
    plot_emergence(series, emergence, args.title)
    plot_comparison(series, comparison, args.comparison_title)
    print(json.dumps({"emergence": str(emergence), "comparison": str(comparison), "series": len(series)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
