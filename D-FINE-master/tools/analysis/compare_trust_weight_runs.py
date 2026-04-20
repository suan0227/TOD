#!/usr/bin/env python3
"""
Compare baseline vs trust-weight runs for the filtered tiny/small protocol.

Outputs:
  - comparison_metrics.png
  - comparison_examples.png
  - comparison_summary.json

The script expects each run directory to contain:
  - log.txt
  - qualitative_analysis/prediction_events.csv
  - qualitative_analysis/gt_events.csv

Example:
  python tools/analysis/compare_trust_weight_runs.py \
    --baseline-dir output/dfine_s_baseline_filtering \
    --trust-dir output/D-Fine_S_filter_trust_weight \
    --output-dir output/trust_weight_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
except ImportError as exc:  # pragma: no cover - user-facing runtime guard
    raise SystemExit(
        "This script needs matplotlib. Install it first, for example: "
        "`pip install matplotlib pillow`."
    ) from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - user-facing runtime guard
    raise SystemExit(
        "This script needs Pillow. Install it first, for example: "
        "`pip install matplotlib pillow`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = REPO_ROOT / "output/dfine_s_baseline_filtering"
DEFAULT_TRUST_DIR = REPO_ROOT / "output/D-Fine_S_filter_trust_weight"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/trust_weight_comparison"

SIZE_BINS = ("tiny", "small")
SCENE_ORDER = ("1-2", "3-5", "6-10", "11+")
METRIC_ORDER = ("mAP@50:95", "AP50", "AP75", "AP_tiny", "AP_s")
METRIC_LABELS = {
    "mAP@50:95": "mAP@50:95",
    "AP50": "AP50",
    "AP75": "AP75",
    "AP_tiny": "AP_tiny",
    "AP_s": "AP_s",
}
LOG_METRICS = {
    "mAP@50:95": ("test_coco_eval_bbox", 0),
    "AP50": ("test_coco_eval_bbox", 1),
    "AP75": ("test_coco_eval_bbox", 2),
    "AP_tiny": ("val/AP_tiny", None),
    "AP_s": ("val/AP_s", None),
}


@dataclass(frozen=True)
class ExamplePair:
    image_id: int
    gt_index: int
    label_id: int
    label_name: str
    size_bin: str
    scene_bin: str
    image_path: Path
    gt_box: Tuple[float, float, float, float]
    baseline_box: Tuple[float, float, float, float]
    trust_box: Tuple[float, float, float, float]
    baseline_iou: float
    trust_iou: float
    baseline_score: float
    trust_score: float

    @property
    def delta_iou(self) -> float:
        return self.trust_iou - self.baseline_iou

    @property
    def delta_score(self) -> float:
        return self.trust_score - self.baseline_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create baseline-vs-trust comparison figures for tiny/small runs."
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=str(DEFAULT_BASELINE_DIR),
        help="Baseline run directory or its qualitative_analysis directory.",
    )
    parser.add_argument(
        "--trust-dir",
        type=str,
        default=str(DEFAULT_TRUST_DIR),
        help="Trust-weight run directory or its qualitative_analysis directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline",
        help="Label used in plots for the baseline run.",
    )
    parser.add_argument(
        "--trust-label",
        type=str,
        default="Trust weight",
        help="Label used in plots for the trust-weight run.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=4,
        help="Number of representative qualitative examples to render.",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=420,
        help="Square panel size for each cropped qualitative example.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=6.0,
        help="Crop side length multiplier around the target GT box.",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=320,
        help="Minimum square crop size in pixels before resizing.",
    )
    parser.add_argument(
        "--scene-order",
        nargs="+",
        default=list(SCENE_ORDER),
        help="Preferred scene-bin order when selecting representative examples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = resolve_run_paths(Path(args.baseline_dir))
    trust = resolve_run_paths(Path(args.trust_dir))
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = extract_best_metrics(baseline.log_path)
    trust_metrics = extract_best_metrics(trust.log_path)

    baseline_preds, _baseline_preds_by_image = load_tp_prediction_rows(baseline.pred_csv)
    trust_preds, _trust_preds_by_image = load_tp_prediction_rows(trust.pred_csv)
    gt_rows, gt_rows_by_image = load_gt_rows(baseline.gt_csv)

    examples = build_example_pairs(
        baseline_preds=baseline_preds,
        trust_preds=trust_preds,
        gt_rows=gt_rows,
    )
    selected_examples = select_examples(
        examples=examples,
        limit=max(1, args.num_examples),
        scene_order=tuple(args.scene_order),
    )

    metrics_path = out_dir / "comparison_metrics.png"
    examples_path = out_dir / "comparison_examples.png"
    summary_path = out_dir / "comparison_summary.json"
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    render_metrics_figure(
        output_path=metrics_path,
        baseline_label=args.baseline_label,
        trust_label=args.trust_label,
        baseline_metrics=baseline_metrics,
        trust_metrics=trust_metrics,
        baseline_tp_rows=[row for row in baseline_preds.values() if row["matched_gt_size_bin"] in SIZE_BINS],
        trust_tp_rows=[row for row in trust_preds.values() if row["matched_gt_size_bin"] in SIZE_BINS],
    )

    render_example_montage(
        output_path=examples_path,
        examples=selected_examples,
        gt_rows_by_image=gt_rows_by_image,
        baseline_label=args.baseline_label,
        trust_label=args.trust_label,
        panel_size=args.panel_size,
        crop_margin=args.crop_margin,
        min_crop_size=args.min_crop_size,
        examples_dir=examples_dir,
    )

    summary = {
        "baseline": {
            "run_dir": str(baseline.run_dir),
            "analysis_dir": str(baseline.analysis_dir),
            "log_path": str(baseline.log_path),
            "pred_csv": str(baseline.pred_csv),
            "gt_csv": str(baseline.gt_csv),
            "best_metrics": baseline_metrics,
        },
        "trust": {
            "run_dir": str(trust.run_dir),
            "analysis_dir": str(trust.analysis_dir),
            "log_path": str(trust.log_path),
            "pred_csv": str(trust.pred_csv),
            "gt_csv": str(trust.gt_csv),
            "best_metrics": trust_metrics,
        },
        "deltas": {
            metric: round(trust_metrics[metric]["value"] - baseline_metrics[metric]["value"], 6)
            for metric in METRIC_ORDER
        },
        "selected_examples": [
            {
                "image_id": ex.image_id,
                "gt_index": ex.gt_index,
                "label_id": ex.label_id,
                "label_name": ex.label_name,
                "size_bin": ex.size_bin,
                "scene_bin": ex.scene_bin,
                "image_path": str(ex.image_path),
                "baseline_iou": ex.baseline_iou,
                "trust_iou": ex.trust_iou,
                "baseline_score": ex.baseline_score,
                "trust_score": ex.trust_score,
                "delta_iou": ex.delta_iou,
                "delta_score": ex.delta_score,
            }
            for ex in selected_examples
        ],
        "outputs": {
            "comparison_metrics": str(metrics_path),
            "comparison_examples": str(examples_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Wrote metrics figure: {metrics_path}")
    print(f"Wrote qualitative montage: {examples_path}")
    print(f"Wrote summary JSON: {summary_path}")
    print("Selected examples:")
    for ex in selected_examples:
        print(
            f"  image {ex.image_id} gt {ex.gt_index} "
            f"({ex.size_bin}, {ex.scene_bin}) "
            f"IoU {ex.baseline_iou:.3f}->{ex.trust_iou:.3f} "
            f"score {ex.baseline_score:.3f}->{ex.trust_score:.3f}"
        )


@dataclass
class RunPaths:
    run_dir: Path
    analysis_dir: Path
    log_path: Path
    pred_csv: Path
    gt_csv: Path


def resolve_run_paths(path: Path) -> RunPaths:
    resolved = resolve_existing_path(path)
    if resolved.name == "qualitative_analysis":
        analysis_dir = resolved
        run_dir = resolved.parent
    elif (resolved / "qualitative_analysis").is_dir():
        analysis_dir = resolved / "qualitative_analysis"
        run_dir = resolved
    else:
        raise FileNotFoundError(
            f"Could not find qualitative_analysis under {resolved}. "
            "Pass either the run directory or the qualitative_analysis directory."
        )

    pred_csv = analysis_dir / "prediction_events.csv"
    gt_csv = analysis_dir / "gt_events.csv"
    log_path = run_dir / "log.txt"
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)
    if not gt_csv.exists():
        raise FileNotFoundError(gt_csv)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    return RunPaths(
        run_dir=run_dir,
        analysis_dir=analysis_dir,
        log_path=log_path,
        pred_csv=pred_csv,
        gt_csv=gt_csv,
    )


def resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    alt = resolve_workspace_path(str(path))
    if alt is not None and Path(alt).exists():
        return Path(alt).resolve()
    if not path.is_absolute():
        repo_path = (REPO_ROOT / path).resolve()
        if repo_path.exists():
            return repo_path
    raise FileNotFoundError(path)


def resolve_workspace_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return path_str
    path = Path(path_str)
    if path.exists():
        return str(path)
    text = str(path)
    if text.startswith("/workspace/suan"):
        alt = "/home/com_2/suan" + text[len("/workspace/suan") :]
        if Path(alt).exists():
            return alt
    if text.startswith("/workspace"):
        alt = "/home/com_2" + text[len("/workspace") :]
        if Path(alt).exists():
            return alt
    return text


def extract_best_metrics(log_path: Path) -> Dict[str, Dict[str, float]]:
    best = {
        metric: {"epoch": -1, "value": float("-inf")}
        for metric in METRIC_ORDER
    }
    with log_path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            epoch = int(row.get("epoch", -1))
            for metric, (source, index) in LOG_METRICS.items():
                value = extract_metric_value(row, source, index)
                if value is None:
                    continue
                if value > best[metric]["value"]:
                    best[metric] = {"epoch": epoch, "value": float(value)}
    return best


def extract_metric_value(row: Dict, source: str, index: Optional[int]) -> Optional[float]:
    if index is None:
        value = row.get(source)
        if value is None:
            return None
        return float(value)
    values = row.get(source)
    if not isinstance(values, list) or len(values) <= index:
        return None
    return float(values[index])


def load_tp_prediction_rows(path: Path) -> Tuple[Dict[Tuple[int, int, int], Dict], Dict[int, List[Dict]]]:
    rows: Dict[Tuple[int, int, int], Dict] = {}
    by_image: Dict[int, List[Dict]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "tp":
                continue
            if not row.get("matched_gt_index"):
                continue
            parsed = {
                "image_id": int(row["image_id"]),
                "matched_gt_index": int(row["matched_gt_index"]),
                "matched_gt_label": int(row["matched_gt_label"]),
                "matched_gt_name": row["matched_gt_name"],
                "matched_gt_size_bin": row["matched_gt_size_bin"],
                "scene_bin": row["scene_bin"],
                "matched_iou": float(row["matched_iou"]),
                "score": float(row["score"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "image_path": Path(resolve_workspace_path(row["image_path"]) or row["image_path"]).resolve(),
            }
            key = (parsed["image_id"], parsed["matched_gt_index"], parsed["matched_gt_label"])
            rows[key] = parsed
            by_image[parsed["image_id"]].append(parsed)
    return rows, by_image


def load_gt_rows(path: Path) -> Tuple[Dict[Tuple[int, int, int], Dict], Dict[int, List[Dict]]]:
    rows: Dict[Tuple[int, int, int], Dict] = {}
    by_image: Dict[int, List[Dict]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("gt_index"):
                continue
            parsed = {
                "image_id": int(row["image_id"]),
                "gt_index": int(row["gt_index"]),
                "label_id": int(row["label_id"]),
                "label_name": row["label_name"],
                "size_bin": row["size_bin"],
                "scene_bin": row["scene_bin"],
                "area": float(row["area"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "image_path": Path(resolve_workspace_path(row["image_path"]) or row["image_path"]).resolve(),
            }
            key = (parsed["image_id"], parsed["gt_index"], parsed["label_id"])
            rows[key] = parsed
            by_image[parsed["image_id"]].append(parsed)
    return rows, by_image


def build_example_pairs(
    baseline_preds: Dict[Tuple[int, int, int], Dict],
    trust_preds: Dict[Tuple[int, int, int], Dict],
    gt_rows: Dict[Tuple[int, int, int], Dict],
) -> List[ExamplePair]:
    examples: List[ExamplePair] = []
    for key, trust_row in trust_preds.items():
        baseline_row = baseline_preds.get(key)
        if baseline_row is None:
            continue
        if baseline_row["matched_gt_size_bin"] not in SIZE_BINS:
            continue
        gt_key = (baseline_row["image_id"], baseline_row["matched_gt_index"], baseline_row["matched_gt_label"])
        gt_row = gt_rows.get(gt_key)
        if gt_row is None:
            continue
        examples.append(
            ExamplePair(
                image_id=baseline_row["image_id"],
                gt_index=baseline_row["matched_gt_index"],
                label_id=baseline_row["matched_gt_label"],
                label_name=baseline_row["matched_gt_name"],
                size_bin=baseline_row["matched_gt_size_bin"],
                scene_bin=baseline_row["scene_bin"],
                image_path=gt_row["image_path"],
                gt_box=(gt_row["x1"], gt_row["y1"], gt_row["x2"], gt_row["y2"]),
                baseline_box=(baseline_row["x1"], baseline_row["y1"], baseline_row["x2"], baseline_row["y2"]),
                trust_box=(trust_row["x1"], trust_row["y1"], trust_row["x2"], trust_row["y2"]),
                baseline_iou=baseline_row["matched_iou"],
                trust_iou=trust_row["matched_iou"],
                baseline_score=baseline_row["score"],
                trust_score=trust_row["score"],
            )
        )
    return examples


def select_examples(
    examples: Sequence[ExamplePair],
    limit: int,
    scene_order: Sequence[str],
) -> List[ExamplePair]:
    ordered = sorted(examples, key=lambda ex: (ex.delta_iou, ex.delta_score), reverse=True)
    by_scene: Dict[str, List[ExamplePair]] = defaultdict(list)
    for ex in ordered:
        by_scene[ex.scene_bin].append(ex)

    selected: List[ExamplePair] = []
    used_images = set()

    for scene in scene_order:
        for ex in by_scene.get(scene, []):
            if ex.image_id in used_images:
                continue
            selected.append(ex)
            used_images.add(ex.image_id)
            break
        if len(selected) >= limit:
            return selected[:limit]

    for ex in ordered:
        if ex.image_id in used_images:
            continue
        selected.append(ex)
        used_images.add(ex.image_id)
        if len(selected) >= limit:
            break

    return selected[:limit]


def render_metrics_figure(
    output_path: Path,
    baseline_label: str,
    trust_label: str,
    baseline_metrics: Dict[str, Dict[str, float]],
    trust_metrics: Dict[str, Dict[str, float]],
    baseline_tp_rows: Sequence[Dict],
    trust_tp_rows: Sequence[Dict],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.patch.set_facecolor("white")

    # Left: best metric comparison.
    ax = axes[0]
    xs = list(range(len(METRIC_ORDER)))
    width = 0.36
    baseline_values = [baseline_metrics[m]["value"] for m in METRIC_ORDER]
    trust_values = [trust_metrics[m]["value"] for m in METRIC_ORDER]
    bars_base = ax.bar([x - width / 2 for x in xs], baseline_values, width, label=baseline_label, color="#9ca3af")
    bars_trust = ax.bar([x + width / 2 for x in xs], trust_values, width, label=trust_label, color="#2563eb")
    ax.set_xticks(xs)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER], rotation=0)
    ax.set_ylim(0, max(max(baseline_values), max(trust_values)) * 1.18)
    ax.set_ylabel("Best validation score")
    ax.set_title("Best checkpoint metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    for bar in list(bars_base) + list(bars_trust):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.003,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    delta_text = "  ".join(
        f"Δ{metric} {trust_metrics[metric]['value'] - baseline_metrics[metric]['value']:+.4f}"
        for metric in METRIC_ORDER
    )
    ax.text(
        0.5,
        -0.18,
        delta_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#374151",
    )

    # Right: tiny/small matched TP IoU survival curve.
    ax = axes[1]
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    baseline_curve = survival_curve(baseline_tp_rows, thresholds)
    trust_curve = survival_curve(trust_tp_rows, thresholds)
    ax.plot(thresholds, baseline_curve, marker="o", lw=2, color="#9ca3af", label=baseline_label)
    ax.plot(thresholds, trust_curve, marker="o", lw=2, color="#2563eb", label=trust_label)
    ax.axvline(0.75, color="#d97706", linestyle="--", lw=1, alpha=0.7)
    ax.set_xlim(0.5, 0.95)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(thresholds)
    ax.set_xlabel("Matched IoU threshold")
    ax.set_ylabel("Fraction of tiny/small TPs above threshold")
    ax.set_title("High-IoU tail on matched tiny/small TPs")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")

    high_iou_note = (
        f"IoU>=0.75: {baseline_curve[5]:.3f} -> {trust_curve[5]:.3f}  |  "
        f"IoU>=0.85: {baseline_curve[7]:.3f} -> {trust_curve[7]:.3f}"
    )
    ax.text(
        0.5,
        -0.18,
        high_iou_note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#374151",
    )

    fig.suptitle(
        "Baseline vs trust-weight comparison on filtered tiny/small AI-TOD",
        fontsize=14,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def survival_curve(rows: Sequence[Dict], thresholds: Sequence[float]) -> List[float]:
    if not rows:
        return [0.0 for _ in thresholds]
    values = [float(row["matched_iou"]) for row in rows if row.get("matched_gt_size_bin") in SIZE_BINS]
    if not values:
        return [0.0 for _ in thresholds]
    total = len(values)
    return [sum(v >= t for v in values) / total for t in thresholds]


def render_example_montage(
    output_path: Path,
    examples: Sequence[ExamplePair],
    gt_rows_by_image: Dict[int, List[Dict]],
    baseline_label: str,
    trust_label: str,
    panel_size: int,
    crop_margin: float,
    min_crop_size: int,
    examples_dir: Path,
) -> None:
    if not examples:
        raise RuntimeError("No overlapping tiny/small TP examples were found.")

    font_path = fm.findfont("DejaVu Sans", fallback_to_default=True)
    title_font = ImageFont.truetype(font_path, size=18)
    caption_font = ImageFont.truetype(font_path, size=15)
    tag_font = ImageFont.truetype(font_path, size=16)

    panel_caption_h = 92
    row_header_h = 44
    row_gap = 18
    col_gap = 18
    margin = 24

    rendered_rows: List[Image.Image] = []
    for idx, ex in enumerate(examples, start=1):
        baseline_panel = render_example_panel(
            ex=ex,
            gt_rows=gt_rows_by_image.get(ex.image_id, []),
            prediction_box=ex.baseline_box,
            panel_title=baseline_label,
            panel_color="#ef4444",
            gt_color="#22c55e",
            target_gt_color="#16a34a",
            panel_size=panel_size,
            crop_margin=crop_margin,
            min_crop_size=min_crop_size,
            caption_lines=[
                f"matched IoU: {ex.baseline_iou:.3f}",
                f"score: {ex.baseline_score:.3f}",
                f"target: {ex.label_name} / {ex.size_bin}",
            ],
            title_font=title_font,
            caption_font=caption_font,
            tag_font=tag_font,
        )
        trust_panel = render_example_panel(
            ex=ex,
            gt_rows=gt_rows_by_image.get(ex.image_id, []),
            prediction_box=ex.trust_box,
            panel_title=trust_label,
            panel_color="#2563eb",
            gt_color="#22c55e",
            target_gt_color="#16a34a",
            panel_size=panel_size,
            crop_margin=crop_margin,
            min_crop_size=min_crop_size,
            caption_lines=[
                f"matched IoU: {ex.trust_iou:.3f}",
                f"score: {ex.trust_score:.3f}",
                f"target: {ex.label_name} / {ex.size_bin}",
            ],
            title_font=title_font,
            caption_font=caption_font,
            tag_font=tag_font,
        )

        row_width = panel_size * 2 + col_gap
        row_height = row_header_h + panel_size + panel_caption_h
        row_canvas = Image.new("RGB", (row_width, row_height), "white")
        row_draw = ImageDraw.Draw(row_canvas)
        row_title = (
            f"Image {ex.image_id} | GT {ex.gt_index} | {ex.label_name} | "
            f"{ex.size_bin} | scene {ex.scene_bin} | "
            f"ΔIoU {ex.delta_iou:+.3f} | Δscore {ex.delta_score:+.3f}"
        )
        row_draw.text((0, 4), row_title, fill="#111827", font=title_font)
        row_canvas.paste(baseline_panel, (0, row_header_h))
        row_canvas.paste(trust_panel, (panel_size + col_gap, row_header_h))
        rendered_rows.append(row_canvas)

        # Save each row as an individual figure for convenient reuse in a paper draft.
        row_path = examples_dir / f"example_{idx:02d}_pair.png"
        row_canvas.save(row_path)

    # Compose full montage.
    montage_width = panel_size * 2 + col_gap + margin * 2
    montage_height = margin * 2 + sum(img.height for img in rendered_rows) + row_gap * (len(rendered_rows) - 1)
    montage = Image.new("RGB", (montage_width, montage_height), "#f8fafc")
    draw = ImageDraw.Draw(montage)
    draw.text((margin, 10), "Qualitative comparison: baseline vs trust weight", fill="#111827", font=title_font)
    draw.text(
        (margin, 34),
        "Green = target GT, red = baseline prediction, blue = trust-weight prediction",
        fill="#475569",
        font=caption_font,
    )

    y = margin + 54
    for row_img in rendered_rows:
        montage.paste(row_img, (margin, y))
        y += row_img.height + row_gap

    montage.save(output_path, dpi=(220, 220))


def render_example_panel(
    ex: ExamplePair,
    gt_rows: Sequence[Dict],
    prediction_box: Tuple[float, float, float, float],
    panel_title: str,
    panel_color: str,
    gt_color: str,
    target_gt_color: str,
    panel_size: int,
    crop_margin: float,
    min_crop_size: int,
    caption_lines: Sequence[str],
    title_font: ImageFont.FreeTypeFont,
    caption_font: ImageFont.FreeTypeFont,
    tag_font: ImageFont.FreeTypeFont,
) -> Image.Image:
    source = Image.open(ex.image_path).convert("RGB")
    crop = square_crop_around_box(
        box=ex.gt_box,
        image_size=source.size,
        margin=crop_margin,
        min_size=min_crop_size,
    )
    crop_img = source.crop(crop)
    crop_side = crop[2] - crop[0]
    scale = panel_size / float(crop_side)
    crop_img = crop_img.resize((panel_size, panel_size), resample=getattr(Image, "Resampling", Image).LANCZOS)

    draw = ImageDraw.Draw(crop_img)
    draw_context_gt_boxes(draw, gt_rows, crop, scale, gt_color, target_key=(ex.image_id, ex.gt_index, ex.label_id))
    clipped_pred = clip_box_to_crop(prediction_box, crop)
    if clipped_pred is not None:
        draw_box(
            draw=draw,
            box=transform_box(clipped_pred, crop, scale),
            outline=panel_color,
            width=4,
            label=None,
            font=caption_font,
        )
    clipped_gt = clip_box_to_crop(ex.gt_box, crop)
    if clipped_gt is not None:
        draw_box(
            draw=draw,
            box=transform_box(clipped_gt, crop, scale),
            outline=target_gt_color,
            width=4,
            label="GT",
            font=caption_font,
        )
    draw_panel_tag(draw, panel_title, panel_color, tag_font)

    caption_h = 92
    panel = Image.new("RGB", (panel_size, panel_size + caption_h), "white")
    panel.paste(crop_img, (0, 0))
    panel_draw = ImageDraw.Draw(panel)
    caption_y = panel_size + 10
    for i, line in enumerate(caption_lines):
        panel_draw.text((10, caption_y + i * 24), line, fill="#111827", font=caption_font)
    panel_draw.line((0, panel_size, panel_size, panel_size), fill="#cbd5e1", width=1)
    return panel


def draw_context_gt_boxes(
    draw: ImageDraw.ImageDraw,
    gt_rows: Sequence[Dict],
    crop: Tuple[int, int, int, int],
    scale: float,
    gt_color: str,
    target_key: Tuple[int, int, int],
) -> None:
    for row in gt_rows:
        key = (row["image_id"], row["gt_index"], row["label_id"])
        clipped = clip_box_to_crop((row["x1"], row["y1"], row["x2"], row["y2"]), crop)
        if clipped is None:
            continue
        if key == target_key:
            continue
        draw_box(
            draw=draw,
            box=transform_box(clipped, crop, scale),
            outline=gt_color,
            width=1,
            label=None,
            font=None,
        )


def draw_panel_tag(draw: ImageDraw.ImageDraw, text: str, fill: str, font: ImageFont.FreeTypeFont) -> None:
    padding_x = 8
    padding_y = 5
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    rect = [8, 8, 8 + text_w + padding_x * 2, 8 + text_h + padding_y * 2]
    draw.rectangle(rect, fill=fill)
    draw.text((rect[0] + padding_x, rect[1] + padding_y - 1), text, fill="white", font=font)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    outline: str,
    width: int,
    label: Optional[str],
    font: Optional[ImageFont.FreeTypeFont],
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
    if label and font is not None:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_w = text_w + 8
        label_h = text_h + 6
        label_x = int(max(0, min(x1, 9999)))
        label_y = int(max(0, y1 - label_h - 1))
        draw.rectangle([label_x, label_y, label_x + label_w, label_y + label_h], fill=outline)
        draw.text((label_x + 4, label_y + 2), label, fill="white", font=font)


def transform_box(
    box: Tuple[float, float, float, float],
    crop: Tuple[int, int, int, int],
    scale: float,
) -> Tuple[float, float, float, float]:
    left, top, _, _ = crop
    x1, y1, x2, y2 = box
    return (
        (x1 - left) * scale,
        (y1 - top) * scale,
        (x2 - left) * scale,
        (y2 - top) * scale,
    )


def clip_box_to_crop(
    box: Tuple[float, float, float, float],
    crop: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float, float, float]]:
    left, top, right, bottom = crop
    x1, y1, x2, y2 = box
    ix1 = max(x1, left)
    iy1 = max(y1, top)
    ix2 = min(x2, right)
    iy2 = min(y2, bottom)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def square_crop_around_box(
    box: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    margin: float,
    min_size: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    img_w, img_h = image_size
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(min_size, int(round(max(bw, bh) * margin)))
    side = min(side, min(img_w, img_h))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = max(0, min(left, img_w - side))
    top = max(0, min(top, img_h - side))
    return left, top, left + side, top + side


if __name__ == "__main__":
    main()
