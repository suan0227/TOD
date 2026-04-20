#!/usr/bin/env python3
"""
Qualitative analysis for D-FINE detection checkpoints.

Outputs:
- summary.json
- per_image.csv
- prediction_events.csv
- gt_events.csv
- confusion_matrix.png / confusion_matrix_normalized.png
- confidence_histogram.png
- iou_distribution.png
- fp_reasons.png / fn_reasons.png
- size_bin_recall.png
- crowded_scene_metrics.png
- score_vs_iou.png
- tsne/tsne_embeddings.csv and optional t-SNE plots
- example overlays and crops
- optional Grad-CAM overlays
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert, box_iou

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import YAMLConfig, yaml_utils
from src.data.dataset.area_filter import DEFAULT_AREA_BUCKETS
from src.misc import dist_utils
from src.solver import TASKS

SIZE_BIN_ORDER = ("very_tiny", "tiny", "small", "medium", "large")
AUTO_CHECKPOINTS = (
    "best_very_tiny.pth",
    "best_stg2.pth",
    "best_stg1.pth",
    "last.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative analysis for D-FINE checkpoints")
    parser.add_argument("--run-dir", type=str, help="Run directory containing checkpoints")
    parser.add_argument("-c", "--config", type=str, help="Config file path")
    parser.add_argument(
        "-r",
        "--checkpoint",
        type=str,
        default="auto",
        help="Checkpoint path or 'auto' when --run-dir is provided",
    )
    parser.add_argument("--output-dir", type=str, help="Analysis output directory")
    parser.add_argument("--val-img-folder", type=str, help="Override validation image folder")
    parser.add_argument("--val-ann-file", type=str, help="Override validation annotation file")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu / ...")
    parser.add_argument("--batch-size", type=int, default=None, help="Override val total batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Override val num_workers")
    parser.add_argument("--max-images", type=int, default=None, help="Limit evaluation images")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--match-iou", type=float, default=0.5, help="TP matching IoU threshold")
    parser.add_argument(
        "--near-iou",
        type=float,
        default=0.1,
        help="IoU threshold used to distinguish nearby localization errors from background errors",
    )
    parser.add_argument(
        "--crowd-bins",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Image GT-count bin edges for crowd analysis",
    )
    parser.add_argument(
        "--crowd-overlap-iou",
        type=float,
        default=0.1,
        help="Pairwise GT IoU threshold used for overlap-crowded subset",
    )
    parser.add_argument(
        "--num-example-images",
        type=int,
        default=8,
        help="Number of scene overlays per category",
    )
    parser.add_argument(
        "--num-example-crops",
        type=int,
        default=12,
        help="Number of FP/FN crops to save",
    )
    parser.add_argument(
        "--save-gradcam",
        action="store_true",
        help="Generate Grad-CAM overlays for representative predictions",
    )
    parser.add_argument(
        "--save-tsne",
        action="store_true",
        help="Generate t-SNE embeddings for representative decoder queries",
    )
    parser.add_argument(
        "--tsne-conf-thresh",
        type=float,
        default=0.0,
        help="Confidence threshold used only for t-SNE sample collection",
    )
    parser.add_argument(
        "--tsne-max-points",
        type=int,
        default=1500,
        help="Maximum number of query points used for t-SNE",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    parser.add_argument(
        "--tsne-iterations",
        type=int,
        default=750,
        help="Number of t-SNE optimization iterations",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate",
    )
    parser.add_argument(
        "--tsne-pca-dim",
        type=int,
        default=50,
        help="Pre-PCA dimension before running t-SNE",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=0,
        help="Random seed for t-SNE subsampling and initialization",
    )
    parser.add_argument(
        "--num-gradcam",
        type=int,
        default=4,
        help="Maximum number of Grad-CAM examples",
    )
    parser.add_argument(
        "--cam-module",
        choices=["backbone_last_return", "backbone_last_stage"],
        default="backbone_last_return",
        help="Module used for Grad-CAM",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, checkpoint_path, run_dir = resolve_inputs(args)
    analysis_dir = resolve_analysis_dir(args, run_dir, checkpoint_path)

    cfg, dataset_override = build_cfg(args, config_path, checkpoint_path, analysis_dir)
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=cfg.seed)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()

    model = solver.ema.module if solver.ema is not None else dist_utils.de_parallel(solver.model)
    model.eval()
    gradcam_model = build_gradcam_model(model)
    tsne_recorder = DecoderInputRecorder(resolve_tsne_module(model)) if args.save_tsne else None
    postprocessor = solver.postprocessor.eval()
    dataset = solver.val_dataloader.dataset
    category_names = build_category_name_map(dataset)
    class_ids = sorted(category_names.keys())
    class_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}
    confusion = np.zeros((len(class_ids) + 1, len(class_ids) + 1), dtype=np.int64)
    tsne_confusion = np.zeros_like(confusion) if args.save_tsne else None

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Analysis output: {analysis_dir}")
    print(f"Validation img folder: {dataset_override['img_folder']}")
    print(f"Validation ann file: {dataset_override['ann_file']}")

    pred_records: List[Dict] = []
    gt_records: List[Dict] = []
    per_image_rows: List[Dict] = []
    tsne_samples: List[Dict] = []

    image_counter = 0
    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            if args.max_images is not None and image_counter >= args.max_images:
                break

            if args.max_images is not None:
                keep = max(0, args.max_images - image_counter)
                targets = targets[:keep]
                samples = samples[:keep]

            samples = samples.to(solver.device)
            targets = [
                {k: v.to(solver.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            outputs = model(samples)
            decoder_inputs = tsne_recorder.pop() if tsne_recorder is not None else None
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            raw_preds = collect_topk_detections(outputs, orig_target_sizes, postprocessor)

            for batch_idx, (target, pred_pack) in enumerate(zip(targets, raw_preds)):
                image_rows = analyze_single_image(
                    target=target,
                    sample=samples[batch_idx],
                    pred_pack=pred_pack,
                    class_to_idx=class_to_idx,
                    confusion=confusion,
                    category_names=category_names,
                    conf_thresh=args.conf_thresh,
                    match_iou=args.match_iou,
                    near_iou=args.near_iou,
                    crowd_bins=sorted(args.crowd_bins),
                    crowd_overlap_iou=args.crowd_overlap_iou,
                )
                per_image_rows.append(image_rows["image"])
                pred_records.extend(image_rows["preds"])
                gt_records.extend(image_rows["gts"])
                if decoder_inputs is not None and batch_idx < decoder_inputs.shape[0]:
                    tsne_image_rows = analyze_single_image(
                        target=target,
                        sample=samples[batch_idx],
                        pred_pack=pred_pack,
                        class_to_idx=class_to_idx,
                        confusion=tsne_confusion,
                        category_names=category_names,
                        conf_thresh=args.tsne_conf_thresh,
                        match_iou=args.match_iou,
                        near_iou=args.near_iou,
                        crowd_bins=sorted(args.crowd_bins),
                        crowd_overlap_iou=args.crowd_overlap_iou,
                    )
                    tsne_samples.extend(
                        build_tsne_samples(
                            pred_rows=tsne_image_rows["preds"],
                            query_embeddings=decoder_inputs[batch_idx],
                        )
                    )
                image_counter += 1

    if tsne_recorder is not None:
        tsne_recorder.close()

    save_analysis_outputs(
        analysis_dir=analysis_dir,
        class_ids=class_ids,
        category_names=category_names,
        confusion=confusion,
        pred_records=pred_records,
        gt_records=gt_records,
        per_image_rows=per_image_rows,
        args=args,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_override=dataset_override,
    )

    save_example_images(
        analysis_dir=analysis_dir,
        pred_records=pred_records,
        gt_records=gt_records,
        per_image_rows=per_image_rows,
        num_scene_examples=args.num_example_images,
        num_crop_examples=args.num_example_crops,
    )

    if args.save_tsne:
        save_tsne_outputs(
            analysis_dir=analysis_dir,
            tsne_samples=tsne_samples,
            class_ids=class_ids,
            category_names=category_names,
            args=args,
        )

    if args.save_gradcam:
        save_gradcam_examples(
            analysis_dir=analysis_dir,
            model=gradcam_model,
            dataset=dataset,
            pred_records=pred_records,
            per_image_rows=per_image_rows,
            device=solver.device,
            cam_module_name=args.cam_module,
            num_gradcam=args.num_gradcam,
        )


def resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path]]:
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)

    if args.config:
        config_path = resolve_existing_path(Path(args.config))
    elif run_dir is not None:
        config_path = infer_config_from_wandb(run_dir)
        if config_path is None:
            raise FileNotFoundError(
                f"Could not infer config from wandb for run dir: {run_dir}. "
                "Pass --config explicitly."
            )
    else:
        raise ValueError("Provide either --config or --run-dir.")

    return config_path, checkpoint_path, run_dir


def resolve_analysis_dir(
    args: argparse.Namespace,
    run_dir: Optional[Path],
    checkpoint_path: Path,
) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    elif run_dir is not None:
        out_dir = run_dir / "qualitative_analysis"
    else:
        out_dir = checkpoint_path.resolve().parent / "qualitative_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def resolve_checkpoint_path(run_dir: Optional[Path], checkpoint_arg: str) -> Path:
    if checkpoint_arg != "auto":
        return resolve_existing_path(Path(checkpoint_arg))
    if run_dir is None:
        raise ValueError("checkpoint='auto' requires --run-dir.")
    for name in AUTO_CHECKPOINTS:
        candidate = run_dir / name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No auto checkpoint found under {run_dir}")


def infer_config_from_wandb(run_dir: Path) -> Optional[Path]:
    wandb_root = REPO_ROOT / "wandb"
    if not wandb_root.exists():
        return None

    target_output = f"./output/{run_dir.name}"
    for config_file in wandb_root.glob("run-*/files/config.yaml"):
        try:
            raw = yaml.safe_load(config_file.read_text())
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        output_dir = raw.get("output_dir", {}).get("value")
        config_value = raw.get("config", {}).get("value")
        if output_dir == target_output and config_value:
            config_path = resolve_existing_path(Path(config_value))
            return config_path
    return None


def build_cfg(
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    analysis_dir: Path,
) -> Tuple[YAMLConfig, Dict[str, str]]:
    raw_cfg = yaml_utils.load_config(str(config_path))
    val_dataset_cfg = raw_cfg["val_dataloader"]["dataset"]

    img_folder = args.val_img_folder or val_dataset_cfg.get("img_folder")
    ann_file = args.val_ann_file or val_dataset_cfg.get("ann_file")
    img_folder = resolve_workspace_path(img_folder)
    ann_file = resolve_workspace_path(ann_file)

    if not Path(img_folder).exists():
        raise FileNotFoundError(f"Validation image folder not found: {img_folder}")
    if not Path(ann_file).exists():
        raise FileNotFoundError(f"Validation annotation file not found: {ann_file}")

    val_overrides = {
        "dataset": {
            "img_folder": img_folder,
            "ann_file": ann_file,
        },
        "num_workers": args.num_workers,
    }
    if args.batch_size is not None:
        val_overrides["total_batch_size"] = args.batch_size

    overrides = {
        "resume": str(checkpoint_path),
        "device": args.device,
        "output_dir": str((analysis_dir / "_runtime").resolve()),
        "use_wandb": False,
        "val_dataloader": val_overrides,
    }
    cfg = YAMLConfig(str(config_path), **overrides)
    return cfg, {"img_folder": img_folder, "ann_file": ann_file}


def build_category_name_map(dataset) -> Dict[int, str]:
    if hasattr(dataset, "category2name"):
        return dataset.category2name
    return {}


def build_gradcam_model(model: torch.nn.Module) -> torch.nn.Module:
    grad_model = copy.deepcopy(model)
    grad_model.eval()
    for param in grad_model.parameters():
        if param.is_floating_point():
            param.requires_grad_(True)
    return grad_model


def resolve_tsne_module(model: torch.nn.Module) -> torch.nn.Module:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise ValueError("Model has no decoder module for t-SNE recording.")
    inner_decoder = getattr(decoder, "decoder", None)
    if inner_decoder is not None:
        return inner_decoder
    return decoder


def collect_topk_detections(
    outputs: Dict[str, torch.Tensor],
    orig_target_sizes: torch.Tensor,
    postprocessor,
) -> List[Dict[str, torch.Tensor]]:
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]

    bbox_pred = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    bbox_pred = bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)

    if postprocessor.use_focal_loss:
        scores = logits.sigmoid()
        scores, flat_index = torch.topk(
            scores.flatten(1), postprocessor.num_top_queries, dim=-1
        )
        class_indices = flat_index % postprocessor.num_classes
        query_indices = flat_index // postprocessor.num_classes
        selected_boxes = bbox_pred.gather(
            dim=1,
            index=query_indices.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]),
        )
        labels = class_indices
    else:
        class_scores = logits.softmax(dim=-1)[:, :, :-1]
        scores, class_indices = class_scores.max(dim=-1)
        query_indices = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(
            scores.shape[0], 1
        )
        if scores.shape[1] > postprocessor.num_top_queries:
            scores, topk_idx = torch.topk(scores, postprocessor.num_top_queries, dim=-1)
            class_indices = class_indices.gather(dim=1, index=topk_idx)
            query_indices = query_indices.gather(dim=1, index=topk_idx)
        selected_boxes = bbox_pred.gather(
            dim=1,
            index=query_indices.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]),
        )
        labels = class_indices

    if getattr(postprocessor, "remap_mscoco_category", False):
        from src.data.dataset import mscoco_label2category

        labels = (
            torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
            .to(labels.device)
            .reshape(labels.shape)
        )

    packed = []
    for image_labels, image_boxes, image_scores, image_queries, image_classes in zip(
        labels, selected_boxes, scores, query_indices, class_indices
    ):
        packed.append(
            {
                "labels": image_labels.detach().cpu(),
                "boxes": image_boxes.detach().cpu(),
                "scores": image_scores.detach().cpu(),
                "query_indices": image_queries.detach().cpu(),
                "class_indices": image_classes.detach().cpu(),
            }
        )
    return packed


def analyze_single_image(
    target: Dict,
    sample: torch.Tensor,
    pred_pack: Dict[str, torch.Tensor],
    class_to_idx: Dict[int, int],
    confusion: np.ndarray,
    category_names: Dict[int, str],
    conf_thresh: float,
    match_iou: float,
    near_iou: float,
    crowd_bins: Sequence[int],
    crowd_overlap_iou: float,
) -> Dict[str, List[Dict]]:
    image_id = int(target["image_id"].item())
    dataset_idx = int(target["idx"].item()) if "idx" in target else image_id
    image_path = str(target["image_path"])
    gt_labels = target["labels"].detach().cpu()
    gt_boxes = resize_boxes_to_original(target["boxes"].detach().cpu(), target["orig_size"], sample.shape[-2:])
    gt_areas = box_areas(gt_boxes)

    raw_scores = pred_pack["scores"]
    raw_labels = pred_pack["labels"]
    raw_boxes = pred_pack["boxes"]
    raw_queries = pred_pack["query_indices"]
    raw_classes = pred_pack["class_indices"]

    keep_mask = raw_scores >= conf_thresh
    kept_indices = keep_mask.nonzero(as_tuple=False).flatten().tolist()

    kept_scores = raw_scores[keep_mask]
    kept_labels = raw_labels[keep_mask]
    kept_boxes = raw_boxes[keep_mask]
    kept_queries = raw_queries[keep_mask]
    kept_classes = raw_classes[keep_mask]

    kept_ious = (
        box_iou(kept_boxes, gt_boxes) if len(kept_boxes) and len(gt_boxes) else torch.zeros((len(kept_boxes), len(gt_boxes)))
    )
    raw_ious = (
        box_iou(raw_boxes, gt_boxes) if len(raw_boxes) and len(gt_boxes) else torch.zeros((len(raw_boxes), len(gt_boxes)))
    )

    matched_pred_local: Dict[int, int] = {}
    matched_gt: Dict[int, int] = {}
    candidates = []
    for pred_local in range(len(kept_boxes)):
        for gt_idx in range(len(gt_boxes)):
            iou = float(kept_ious[pred_local, gt_idx].item())
            if iou >= match_iou:
                candidates.append((iou, float(kept_scores[pred_local].item()), pred_local, gt_idx))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    for iou, _, pred_local, gt_idx in candidates:
        if pred_local in matched_pred_local or gt_idx in matched_gt:
            continue
        matched_pred_local[pred_local] = gt_idx
        matched_gt[gt_idx] = pred_local

    num_gt = len(gt_boxes)
    pair_overlap_count = count_overlapping_pairs(gt_boxes, crowd_overlap_iou)
    mean_nn_distance = mean_nearest_center_distance(gt_boxes)
    scene_bin = crowd_bin_label(num_gt, crowd_bins)
    overlap_crowded = pair_overlap_count > 0

    pred_rows: List[Dict] = []
    gt_rows: List[Dict] = []
    gt_status_map: Dict[int, str] = {}

    for pred_local, gt_idx in matched_pred_local.items():
        pred_label = int(kept_labels[pred_local].item())
        gt_label = int(gt_labels[gt_idx].item())
        pred_box = kept_boxes[pred_local]
        gt_box = gt_boxes[gt_idx]
        match_iou_value = float(kept_ious[pred_local, gt_idx].item())
        pred_base = build_pred_base_row(
            image_id=image_id,
            dataset_idx=dataset_idx,
            image_path=image_path,
            pred_index=kept_indices[pred_local],
            pred_local_index=pred_local,
            label=pred_label,
            score=float(kept_scores[pred_local].item()),
            box=pred_box,
            query_idx=int(kept_queries[pred_local].item()),
            class_idx=int(kept_classes[pred_local].item()),
            category_names=category_names,
            scene_bin=scene_bin,
            num_gt=num_gt,
            pair_overlap_count=pair_overlap_count,
        )
        gt_base = build_gt_base_row(
            image_id=image_id,
            dataset_idx=dataset_idx,
            image_path=image_path,
            gt_index=gt_idx,
            label=gt_label,
            box=gt_box,
            area=float(gt_areas[gt_idx].item()),
            category_names=category_names,
            scene_bin=scene_bin,
            num_gt=num_gt,
            pair_overlap_count=pair_overlap_count,
        )

        confusion[class_to_idx[gt_label], class_to_idx[pred_label]] += 1
        if pred_label == gt_label:
            pred_rows.append(
                {
                    **pred_base,
                    "status": "tp",
                    "reason": "tp",
                    "matched_gt_index": gt_idx,
                    "matched_gt_label": gt_label,
                    "matched_gt_name": category_names.get(gt_label, str(gt_label)),
                    "matched_iou": match_iou_value,
                    "best_any_iou": match_iou_value,
                    "best_same_iou": match_iou_value,
                    "matched_gt_area": float(gt_areas[gt_idx].item()),
                    "matched_gt_size_bin": area_to_size_bin(float(gt_areas[gt_idx].item())),
                }
            )
            gt_rows.append(
                {
                    **gt_base,
                    "status": "tp",
                    "reason": "tp",
                    "matched_pred_index": kept_indices[pred_local],
                    "matched_pred_label": pred_label,
                    "matched_pred_name": category_names.get(pred_label, str(pred_label)),
                    "matched_pred_score": float(kept_scores[pred_local].item()),
                    "matched_iou": match_iou_value,
                }
            )
            gt_status_map[gt_idx] = "tp"
        else:
            pred_rows.append(
                {
                    **pred_base,
                    "status": "fp",
                    "reason": "misclassification",
                    "matched_gt_index": gt_idx,
                    "matched_gt_label": gt_label,
                    "matched_gt_name": category_names.get(gt_label, str(gt_label)),
                    "matched_iou": match_iou_value,
                    "best_any_iou": match_iou_value,
                    "best_same_iou": best_same_iou(raw_ious, keep_mask, gt_labels, pred_label, kept_indices[pred_local]),
                }
            )
            gt_rows.append(
                {
                    **gt_base,
                    "status": "fn",
                    "reason": "misclassification",
                    "matched_pred_index": kept_indices[pred_local],
                    "matched_pred_label": pred_label,
                    "matched_pred_name": category_names.get(pred_label, str(pred_label)),
                    "matched_pred_score": float(kept_scores[pred_local].item()),
                    "matched_iou": match_iou_value,
                }
            )
            gt_status_map[gt_idx] = "fn"

    for pred_local, raw_idx in enumerate(kept_indices):
        if pred_local in matched_pred_local:
            continue
        pred_label = int(kept_labels[pred_local].item())
        reason, extra = classify_unmatched_prediction(
            pred_local=pred_local,
            raw_index=raw_idx,
            pred_label=pred_label,
            gt_labels=gt_labels,
            kept_ious=kept_ious,
            matched_gt=matched_gt,
            match_iou=match_iou,
            near_iou=near_iou,
        )
        confusion[-1, class_to_idx[pred_label]] += 1
        pred_rows.append(
            {
                **build_pred_base_row(
                    image_id=image_id,
                    dataset_idx=dataset_idx,
                    image_path=image_path,
                    pred_index=raw_idx,
                    pred_local_index=pred_local,
                    label=pred_label,
                    score=float(kept_scores[pred_local].item()),
                    box=kept_boxes[pred_local],
                    query_idx=int(kept_queries[pred_local].item()),
                    class_idx=int(kept_classes[pred_local].item()),
                    category_names=category_names,
                    scene_bin=scene_bin,
                    num_gt=num_gt,
                    pair_overlap_count=pair_overlap_count,
                ),
                "status": "fp",
                "reason": reason,
                **extra,
            }
        )

    for gt_idx in range(len(gt_boxes)):
        if gt_idx in gt_status_map:
            continue
        gt_label = int(gt_labels[gt_idx].item())
        reason, extra = classify_unmatched_gt(
            gt_idx=gt_idx,
            gt_label=gt_label,
            gt_area=float(gt_areas[gt_idx].item()),
            gt_labels=gt_labels,
            kept_scores=kept_scores,
            kept_labels=kept_labels,
            kept_indices=kept_indices,
            kept_ious=kept_ious,
            raw_scores=raw_scores,
            raw_labels=raw_labels,
            raw_ious=raw_ious,
            conf_thresh=conf_thresh,
            match_iou=match_iou,
            near_iou=near_iou,
            category_names=category_names,
        )
        confusion[class_to_idx[gt_label], -1] += 1
        gt_rows.append(
            {
                **build_gt_base_row(
                    image_id=image_id,
                    dataset_idx=dataset_idx,
                    image_path=image_path,
                    gt_index=gt_idx,
                    label=gt_label,
                    box=gt_boxes[gt_idx],
                    area=float(gt_areas[gt_idx].item()),
                    category_names=category_names,
                    scene_bin=scene_bin,
                    num_gt=num_gt,
                    pair_overlap_count=pair_overlap_count,
                ),
                "status": "fn",
                "reason": reason,
                **extra,
            }
        )

    image_row = {
        "image_id": image_id,
        "dataset_idx": dataset_idx,
        "image_path": image_path,
        "num_gt": num_gt,
        "num_predictions": int(len(kept_indices)),
        "num_tp": sum(1 for row in pred_rows if row["status"] == "tp"),
        "num_fp": sum(1 for row in pred_rows if row["status"] == "fp"),
        "num_fn": sum(1 for row in gt_rows if row["status"] == "fn"),
        "precision": safe_div(
            sum(1 for row in pred_rows if row["status"] == "tp"),
            len(pred_rows),
        ),
        "recall": safe_div(
            sum(1 for row in gt_rows if row["status"] == "tp"),
            len(gt_rows),
        ),
        "scene_bin": scene_bin,
        "pair_overlap_count": pair_overlap_count,
        "overlap_crowded": overlap_crowded,
        "mean_nearest_center_distance": mean_nn_distance,
    }
    for size_bin in SIZE_BIN_ORDER:
        image_row[f"num_gt_{size_bin}"] = sum(
            1 for row in gt_rows if row["size_bin"] == size_bin
        )
        image_row[f"num_fn_{size_bin}"] = sum(
            1 for row in gt_rows if row["size_bin"] == size_bin and row["status"] == "fn"
        )

    return {"image": image_row, "preds": pred_rows, "gts": gt_rows}


def build_pred_base_row(
    image_id: int,
    dataset_idx: int,
    image_path: str,
    pred_index: int,
    pred_local_index: int,
    label: int,
    score: float,
    box: torch.Tensor,
    query_idx: int,
    class_idx: int,
    category_names: Dict[int, str],
    scene_bin: str,
    num_gt: int,
    pair_overlap_count: int,
) -> Dict:
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    pred_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return {
        "image_id": image_id,
        "dataset_idx": dataset_idx,
        "image_path": image_path,
        "pred_index": pred_index,
        "pred_local_index": pred_local_index,
        "label_id": label,
        "label_name": category_names.get(label, str(label)),
        "score": score,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "pred_area": pred_area,
        "pred_size_bin": area_to_size_bin(pred_area),
        "query_idx": query_idx,
        "class_idx": class_idx,
        "scene_bin": scene_bin,
        "num_gt": num_gt,
        "pair_overlap_count": pair_overlap_count,
    }


def build_gt_base_row(
    image_id: int,
    dataset_idx: int,
    image_path: str,
    gt_index: int,
    label: int,
    box: torch.Tensor,
    area: float,
    category_names: Dict[int, str],
    scene_bin: str,
    num_gt: int,
    pair_overlap_count: int,
) -> Dict:
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    return {
        "image_id": image_id,
        "dataset_idx": dataset_idx,
        "image_path": image_path,
        "gt_index": gt_index,
        "label_id": label,
        "label_name": category_names.get(label, str(label)),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "area": area,
        "size_bin": area_to_size_bin(area),
        "scene_bin": scene_bin,
        "num_gt": num_gt,
        "pair_overlap_count": pair_overlap_count,
    }


def classify_unmatched_prediction(
    pred_local: int,
    raw_index: int,
    pred_label: int,
    gt_labels: torch.Tensor,
    kept_ious: torch.Tensor,
    matched_gt: Dict[int, int],
    match_iou: float,
    near_iou: float,
) -> Tuple[str, Dict]:
    if kept_ious.numel() == 0 or kept_ious.shape[1] == 0:
        return "background", {"best_any_iou": 0.0, "best_same_iou": 0.0}

    overlaps = kept_ious[pred_local]
    best_any_iou, best_any_gt_idx = max_iou_and_index(overlaps)
    same_mask = gt_labels == pred_label
    best_same_iou, best_same_gt_idx = max_iou_and_index(overlaps[same_mask])
    best_same_gt_global = masked_index_to_global(same_mask, best_same_gt_idx)

    if best_same_gt_global is not None and best_same_iou >= match_iou and best_same_gt_global in matched_gt:
        reason = "duplicate"
    elif best_any_iou >= match_iou and int(gt_labels[best_any_gt_idx].item()) != pred_label:
        reason = "class_confusion_overlap"
    elif best_same_iou >= near_iou:
        reason = "localization"
    elif best_any_iou >= near_iou:
        reason = "near_object"
    else:
        reason = "background"

    extra = {
        "best_any_iou": best_any_iou,
        "best_any_gt_index": best_any_gt_idx,
        "best_any_gt_label": int(gt_labels[best_any_gt_idx].item()) if best_any_gt_idx is not None else None,
        "best_same_iou": best_same_iou,
        "best_same_gt_index": best_same_gt_global,
    }
    return reason, extra


def classify_unmatched_gt(
    gt_idx: int,
    gt_label: int,
    gt_area: float,
    gt_labels: torch.Tensor,
    kept_scores: torch.Tensor,
    kept_labels: torch.Tensor,
    kept_indices: List[int],
    kept_ious: torch.Tensor,
    raw_scores: torch.Tensor,
    raw_labels: torch.Tensor,
    raw_ious: torch.Tensor,
    conf_thresh: float,
    match_iou: float,
    near_iou: float,
    category_names: Dict[int, str],
) -> Tuple[str, Dict]:
    kept_col = kept_ious[:, gt_idx] if kept_ious.numel() else torch.tensor([])
    raw_col = raw_ious[:, gt_idx] if raw_ious.numel() else torch.tensor([])

    best_kept_any_iou, best_kept_any_idx = max_iou_and_index(kept_col)
    same_kept_mask = kept_labels == gt_label if len(kept_labels) else torch.tensor([], dtype=torch.bool)
    best_kept_same_iou, best_kept_same_idx = max_iou_and_index(kept_col[same_kept_mask]) if len(kept_col) else (0.0, None)
    best_kept_same_global = masked_index_to_global(same_kept_mask, best_kept_same_idx)

    best_raw_any_iou, best_raw_any_idx = max_iou_and_index(raw_col)
    same_raw_mask = raw_labels == gt_label if len(raw_labels) else torch.tensor([], dtype=torch.bool)
    best_raw_same_iou, best_raw_same_idx = max_iou_and_index(raw_col[same_raw_mask]) if len(raw_col) else (0.0, None)
    best_raw_same_global = masked_index_to_global(same_raw_mask, best_raw_same_idx)

    reason = "missed"
    if best_kept_any_idx is not None and best_kept_any_iou >= match_iou:
        pred_label = int(kept_labels[best_kept_any_idx].item())
        if pred_label != gt_label:
            reason = "class_confusion_overlap"
    elif best_kept_same_iou >= near_iou:
        reason = "localization"
    elif best_raw_same_global is not None and best_raw_same_iou >= match_iou:
        if float(raw_scores[best_raw_same_global].item()) < conf_thresh:
            reason = "low_confidence"
    elif best_raw_same_global is not None and best_raw_same_iou >= near_iou:
        if float(raw_scores[best_raw_same_global].item()) < conf_thresh:
            reason = "low_confidence_localization"
    elif best_raw_any_idx is not None and best_raw_any_iou >= match_iou:
        raw_any_label = int(raw_labels[best_raw_any_idx].item())
        if raw_any_label != gt_label and float(raw_scores[best_raw_any_idx].item()) < conf_thresh:
            reason = "low_confidence_wrong_class"
    elif best_raw_any_iou >= near_iou:
        reason = "missed_nearby"

    extra = {
        "matched_pred_index": kept_indices[best_kept_any_idx] if best_kept_any_idx is not None and best_kept_any_idx < len(kept_indices) else None,
        "matched_pred_label": int(kept_labels[best_kept_any_idx].item()) if best_kept_any_idx is not None else None,
        "matched_pred_name": category_names.get(int(kept_labels[best_kept_any_idx].item()), str(int(kept_labels[best_kept_any_idx].item()))) if best_kept_any_idx is not None else None,
        "matched_pred_score": float(kept_scores[best_kept_any_idx].item()) if best_kept_any_idx is not None else None,
        "best_filtered_any_iou": best_kept_any_iou,
        "best_filtered_same_iou": best_kept_same_iou,
        "best_raw_any_iou": best_raw_any_iou,
        "best_raw_any_score": float(raw_scores[best_raw_any_idx].item()) if best_raw_any_idx is not None else None,
        "best_raw_same_iou": best_raw_same_iou,
        "best_raw_same_score": float(raw_scores[best_raw_same_global].item()) if best_raw_same_global is not None else None,
        "gt_area": gt_area,
    }
    return reason, extra


def best_same_iou(
    raw_ious: torch.Tensor,
    keep_mask: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_label: int,
    raw_idx: int,
) -> float:
    if raw_ious.numel() == 0 or raw_idx >= raw_ious.shape[0]:
        return 0.0
    same_mask = gt_labels == pred_label
    if not bool(same_mask.any()):
        return 0.0
    return float(raw_ious[raw_idx][same_mask].max().item())


def resize_boxes_to_original(
    boxes: torch.Tensor,
    orig_size: torch.Tensor,
    resized_hw: Sequence[int],
) -> torch.Tensor:
    orig_w, orig_h = int(orig_size[0].item()), int(orig_size[1].item())
    resized_h, resized_w = int(resized_hw[0]), int(resized_hw[1])
    scaled = boxes.clone().float()
    if resized_w > 0:
        scaled[:, [0, 2]] *= orig_w / resized_w
    if resized_h > 0:
        scaled[:, [1, 3]] *= orig_h / resized_h
    return scaled


def box_areas(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def area_to_size_bin(area: float) -> str:
    for name in SIZE_BIN_ORDER:
        lower, upper = DEFAULT_AREA_BUCKETS[name]
        if lower <= area < upper:
            return name
    return "large"


def count_overlapping_pairs(boxes: torch.Tensor, iou_thresh: float) -> int:
    if len(boxes) < 2:
        return 0
    pair_ious = box_iou(boxes, boxes)
    count = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if float(pair_ious[i, j].item()) >= iou_thresh:
                count += 1
    return count


def mean_nearest_center_distance(boxes: torch.Tensor) -> float:
    if len(boxes) < 2:
        return 0.0
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    distances = torch.cdist(centers, centers, p=2)
    distances += torch.eye(len(boxes)) * 1e6
    return float(distances.min(dim=1).values.mean().item())


def crowd_bin_label(num_gt: int, thresholds: Sequence[int]) -> str:
    if num_gt == 0:
        return "0"
    lower = 1
    for threshold in thresholds:
        if num_gt <= threshold:
            return f"{lower}-{threshold}"
        lower = threshold + 1
    return f"{lower}+"


def scene_bin_sort_key(label: str) -> Tuple[int, int]:
    if label == "0":
        return (0, 0)
    if label.endswith("+"):
        start = int(label[:-1])
        return (start, 10**9)
    start, end = label.split("-", 1)
    return (int(start), int(end))


def max_iou_and_index(values: torch.Tensor) -> Tuple[float, Optional[int]]:
    if values.numel() == 0:
        return 0.0, None
    max_value, max_index = values.max(dim=0)
    return float(max_value.item()), int(max_index.item())


def masked_index_to_global(mask: torch.Tensor, masked_index: Optional[int]) -> Optional[int]:
    if masked_index is None or mask.numel() == 0:
        return None
    global_indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if masked_index >= len(global_indices):
        return None
    return global_indices[masked_index]


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def save_analysis_outputs(
    analysis_dir: Path,
    class_ids: Sequence[int],
    category_names: Dict[int, str],
    confusion: np.ndarray,
    pred_records: List[Dict],
    gt_records: List[Dict],
    per_image_rows: List[Dict],
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    dataset_override: Dict[str, str],
) -> None:
    summary = build_summary(
        pred_records=pred_records,
        gt_records=gt_records,
        per_image_rows=per_image_rows,
        args=args,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_override=dataset_override,
    )

    write_json(analysis_dir / "summary.json", summary)
    write_csv(analysis_dir / "per_image.csv", per_image_rows)
    write_csv(analysis_dir / "prediction_events.csv", pred_records)
    write_csv(analysis_dir / "gt_events.csv", gt_records)

    labels = [category_names.get(class_id, str(class_id)) for class_id in class_ids] + ["background"]
    plot_confusion_matrix(confusion, labels, analysis_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(
        confusion,
        labels,
        analysis_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_confidence_histogram(pred_records, analysis_dir / "confidence_histogram.png")
    plot_iou_distribution(
        pred_records=pred_records,
        gt_records=gt_records,
        out_path=analysis_dir / "iou_distribution.png",
    )
    plot_reason_counts(pred_records, "fp", analysis_dir / "fp_reasons.png")
    plot_reason_counts(gt_records, "fn", analysis_dir / "fn_reasons.png")
    plot_size_bin_recall(gt_records, analysis_dir / "size_bin_recall.png")
    plot_crowded_scene_metrics(per_image_rows, analysis_dir / "crowded_scene_metrics.png")
    plot_score_vs_iou(pred_records, analysis_dir / "score_vs_iou.png")


def build_summary(
    pred_records: List[Dict],
    gt_records: List[Dict],
    per_image_rows: List[Dict],
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    dataset_override: Dict[str, str],
) -> Dict:
    tp = sum(1 for row in pred_records if row["status"] == "tp")
    fp = sum(1 for row in pred_records if row["status"] == "fp")
    fn = sum(1 for row in gt_records if row["status"] == "fn")
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    fp_reasons = Counter(row["reason"] for row in pred_records if row["status"] == "fp")
    fn_reasons = Counter(row["reason"] for row in gt_records if row["status"] == "fn")

    size_bin_stats = {}
    for size_bin in SIZE_BIN_ORDER:
        tp_count = sum(1 for row in gt_records if row["size_bin"] == size_bin and row["status"] == "tp")
        fn_count = sum(1 for row in gt_records if row["size_bin"] == size_bin and row["status"] == "fn")
        size_bin_stats[size_bin] = {
            "tp": tp_count,
            "fn": fn_count,
            "recall": safe_div(tp_count, tp_count + fn_count),
        }

    class_stats = {}
    label_ids = sorted({row["label_id"] for row in gt_records} | {row["label_id"] for row in pred_records})
    for label_id in label_ids:
        cls_tp = sum(1 for row in pred_records if row["label_id"] == label_id and row["status"] == "tp")
        cls_fp = sum(1 for row in pred_records if row["label_id"] == label_id and row["status"] == "fp")
        cls_fn = sum(1 for row in gt_records if row["label_id"] == label_id and row["status"] == "fn")
        class_stats[str(label_id)] = {
            "precision": safe_div(cls_tp, cls_tp + cls_fp),
            "recall": safe_div(cls_tp, cls_tp + cls_fn),
            "tp": cls_tp,
            "fp": cls_fp,
            "fn": cls_fn,
        }

    crowd_stats = {}
    scene_bins = sorted({row["scene_bin"] for row in per_image_rows}, key=scene_bin_sort_key)
    for scene_bin in scene_bins:
        scene_rows = [row for row in per_image_rows if row["scene_bin"] == scene_bin]
        crowd_stats[scene_bin] = {
            "num_images": len(scene_rows),
            "mean_precision": float(np.mean([row["precision"] for row in scene_rows])) if scene_rows else 0.0,
            "mean_recall": float(np.mean([row["recall"] for row in scene_rows])) if scene_rows else 0.0,
            "mean_fp": float(np.mean([row["num_fp"] for row in scene_rows])) if scene_rows else 0.0,
            "mean_fn": float(np.mean([row["num_fn"] for row in scene_rows])) if scene_rows else 0.0,
        }

    overlap_rows = [row for row in per_image_rows if row["overlap_crowded"]]
    non_overlap_rows = [row for row in per_image_rows if not row["overlap_crowded"]]

    return {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "val_img_folder": dataset_override["img_folder"],
        "val_ann_file": dataset_override["ann_file"],
        "num_images": len(per_image_rows),
        "conf_thresh": args.conf_thresh,
        "match_iou": args.match_iou,
        "near_iou": args.near_iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_reasons": dict(fp_reasons),
        "fn_reasons": dict(fn_reasons),
        "size_bin_recall": size_bin_stats,
        "class_metrics": class_stats,
        "crowded_scene_by_count": crowd_stats,
        "overlap_crowded_subset": {
            "num_images": len(overlap_rows),
            "mean_precision": float(np.mean([row["precision"] for row in overlap_rows])) if overlap_rows else 0.0,
            "mean_recall": float(np.mean([row["recall"] for row in overlap_rows])) if overlap_rows else 0.0,
        },
        "non_overlap_subset": {
            "num_images": len(non_overlap_rows),
            "mean_precision": float(np.mean([row["precision"] for row in non_overlap_rows])) if non_overlap_rows else 0.0,
            "mean_recall": float(np.mean([row["recall"] for row in non_overlap_rows])) if non_overlap_rows else 0.0,
        },
        "notes": [
            "crowded scene metrics are based on GT-count bins and GT pair overlap, not COCO iscrowd flags",
            "FP/FN reasons are heuristic post-hoc labels derived from IoU and confidence patterns",
            "t-SNE, when enabled, is built from decoder query embeddings after deduplicating focal-loss class projections",
            "Grad-CAM overlays are generated on the resized model input",
        ],
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    out_path: Path,
    normalize: bool,
) -> None:
    plt.figure(figsize=(max(8, len(labels) * 0.9), max(6, len(labels) * 0.8)))
    plot_matrix = matrix.astype(np.float64)
    if normalize:
        row_sums = plot_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        plot_matrix = plot_matrix / row_sums
    plt.imshow(plot_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    thresh = plot_matrix.max() / 2.0 if plot_matrix.size else 0.0
    for i in range(plot_matrix.shape[0]):
        for j in range(plot_matrix.shape[1]):
            value = plot_matrix[i, j]
            if normalize:
                text = f"{value:.2f}"
            else:
                text = str(int(value))
            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confidence_histogram(pred_records: List[Dict], out_path: Path) -> None:
    tp_scores = [row["score"] for row in pred_records if row["status"] == "tp"]
    fp_scores = [row["score"] for row in pred_records if row["status"] == "fp"]
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(8, 5))
    if tp_scores:
        plt.hist(tp_scores, bins=bins, alpha=0.6, label="TP", color="#2b6cb0")
    if fp_scores:
        plt.hist(fp_scores, bins=bins, alpha=0.6, label="FP", color="#c53030")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Histogram")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_iou_distribution(pred_records: List[Dict], gt_records: List[Dict], out_path: Path) -> None:
    tp_ious = [row["matched_iou"] for row in pred_records if row["status"] == "tp"]
    fp_ious = [row.get("best_any_iou", 0.0) for row in pred_records if row["status"] == "fp"]
    fn_ious = [row.get("best_raw_same_iou", row.get("best_filtered_same_iou", 0.0)) for row in gt_records if row["status"] == "fn"]
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(8, 5))
    if tp_ious:
        plt.hist(tp_ious, bins=bins, alpha=0.55, label="TP matched IoU", color="#2f855a")
    if fp_ious:
        plt.hist(fp_ious, bins=bins, alpha=0.45, label="FP best IoU", color="#dd6b20")
    if fn_ious:
        plt.hist(fn_ious, bins=bins, alpha=0.45, label="FN best IoU", color="#805ad5")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.title("IoU Distribution")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_reason_counts(rows: List[Dict], status: str, out_path: Path) -> None:
    counter = Counter(row["reason"] for row in rows if row["status"] == status)
    if not counter:
        return
    labels, values = zip(*counter.most_common())
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, color="#4a5568")
    plt.ylabel("Count")
    plt.title(f"{status.upper()} Reasons")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_size_bin_recall(gt_records: List[Dict], out_path: Path) -> None:
    recalls = []
    counts = []
    for size_bin in SIZE_BIN_ORDER:
        tp_count = sum(1 for row in gt_records if row["size_bin"] == size_bin and row["status"] == "tp")
        fn_count = sum(1 for row in gt_records if row["size_bin"] == size_bin and row["status"] == "fn")
        recalls.append(safe_div(tp_count, tp_count + fn_count))
        counts.append(tp_count + fn_count)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(SIZE_BIN_ORDER, recalls, color="#2b6cb0")
    plt.ylim(0, 1)
    plt.ylabel("Recall")
    plt.title("Recall by Size Bin")
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_crowded_scene_metrics(per_image_rows: List[Dict], out_path: Path) -> None:
    scene_bins = sorted({row["scene_bin"] for row in per_image_rows}, key=scene_bin_sort_key)
    recalls = [float(np.mean([row["recall"] for row in per_image_rows if row["scene_bin"] == scene_bin])) for scene_bin in scene_bins]
    precisions = [float(np.mean([row["precision"] for row in per_image_rows if row["scene_bin"] == scene_bin])) for scene_bin in scene_bins]
    counts = [sum(1 for row in per_image_rows if row["scene_bin"] == scene_bin) for scene_bin in scene_bins]

    x = np.arange(len(scene_bins))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - width / 2, recalls, width, label="Recall", color="#2f855a")
    ax1.bar(x + width / 2, precisions, width, label="Precision", color="#2b6cb0")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Metric")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scene_bins)
    ax1.set_xlabel("GT-count scene bin")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, counts, color="#c53030", marker="o", label="Images")
    ax2.set_ylabel("Images")
    ax2.legend(loc="upper right")
    plt.title("Crowded Scene Metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_score_vs_iou(pred_records: List[Dict], out_path: Path) -> None:
    tp_points = [
        (row["score"], row.get("matched_iou", row.get("best_any_iou", 0.0)))
        for row in pred_records
        if row["status"] == "tp"
    ]
    fp_points = [
        (row["score"], row.get("best_any_iou", 0.0))
        for row in pred_records
        if row["status"] == "fp"
    ]
    plt.figure(figsize=(8, 5))
    if tp_points:
        tp_x, tp_y = zip(*tp_points)
        plt.scatter(tp_x, tp_y, alpha=0.5, s=12, label="TP", color="#2f855a")
    if fp_points:
        fp_x, fp_y = zip(*fp_points)
        plt.scatter(fp_x, fp_y, alpha=0.5, s=12, label="FP", color="#c53030")
    plt.xlabel("Confidence")
    plt.ylabel("IoU")
    plt.title("Confidence vs IoU")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify_csv_value(row.get(key)) for key in fieldnames})


def stringify_csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_example_images(
    analysis_dir: Path,
    pred_records: List[Dict],
    gt_records: List[Dict],
    per_image_rows: List[Dict],
    num_scene_examples: int,
    num_crop_examples: int,
) -> None:
    image_to_preds = group_rows(pred_records, "image_id")
    image_to_gts = group_rows(gt_records, "image_id")

    scenes_dir = analysis_dir / "examples" / "scene_overlays"
    crops_dir = analysis_dir / "examples" / "crops"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    top_fn_images = sorted(
        [row for row in per_image_rows if row["num_fn"] > 0],
        key=lambda row: (-row["num_fn"], -row["num_gt"], row["image_id"]),
    )[:num_scene_examples]
    top_fp_images = sorted(
        [row for row in per_image_rows if row["num_fp"] > 0],
        key=lambda row: (-row["num_fp"], -row["num_predictions"], row["image_id"]),
    )[:num_scene_examples]
    top_crowded_images = sorted(
        [row for row in per_image_rows if row["pair_overlap_count"] > 0 or row["num_gt"] >= 6],
        key=lambda row: (-row["num_gt"], row["recall"], row["image_id"]),
    )[:num_scene_examples]

    save_scene_list(top_fn_images, image_to_preds, image_to_gts, scenes_dir / "top_fn")
    save_scene_list(top_fp_images, image_to_preds, image_to_gts, scenes_dir / "top_fp")
    save_scene_list(top_crowded_images, image_to_preds, image_to_gts, scenes_dir / "crowded")

    fp_crops = sorted(
        [row for row in pred_records if row["status"] == "fp"],
        key=lambda row: (-row["score"], -row.get("best_any_iou", 0.0), row["image_id"]),
    )[:num_crop_examples]
    fn_crops = sorted(
        [row for row in gt_records if row["status"] == "fn"],
        key=lambda row: (row["area"], row["image_id"], row["gt_index"]),
    )[:num_crop_examples]

    save_crop_list(
        fp_crops,
        image_to_preds=image_to_preds,
        image_to_gts=image_to_gts,
        out_dir=crops_dir / "fp",
        focus_kind="pred",
    )
    save_crop_list(
        fn_crops,
        image_to_preds=image_to_preds,
        image_to_gts=image_to_gts,
        out_dir=crops_dir / "fn",
        focus_kind="gt",
    )


def group_rows(rows: Iterable[Dict], key: str) -> Dict[int, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def save_scene_list(
    scene_rows: List[Dict],
    image_to_preds: Dict[int, List[Dict]],
    image_to_gts: Dict[int, List[Dict]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, scene in enumerate(scene_rows):
        image_id = scene["image_id"]
        render_scene_overlay(
            image_path=scene["image_path"],
            gt_rows=image_to_gts.get(image_id, []),
            pred_rows=image_to_preds.get(image_id, []),
            out_path=out_dir / f"{rank:02d}_img{image_id}_p{scene['num_fp']}_f{scene['num_fn']}.png",
        )


def save_crop_list(
    rows: List[Dict],
    image_to_preds: Dict[int, List[Dict]],
    image_to_gts: Dict[int, List[Dict]],
    out_dir: Path,
    focus_kind: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows):
        image_id = row["image_id"]
        box = row_box(row)
        render_crop(
            image_path=row["image_path"],
            gt_rows=image_to_gts.get(image_id, []),
            pred_rows=image_to_preds.get(image_id, []),
            focus_box=box,
            out_path=out_dir / f"{rank:02d}_img{image_id}_{focus_kind}_{row['reason']}.png",
        )


def render_scene_overlay(
    image_path: str,
    gt_rows: List[Dict],
    pred_rows: List[Dict],
    out_path: Path,
    max_side: int = 1600,
) -> None:
    image = Image.open(image_path).convert("RGB")
    image, scale = resize_for_display(image, max_side=max_side)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for row in gt_rows:
        draw_box(draw, scale_box(row_box(row), scale), gt_color(row), f"GT:{row['label_name']}", font)
    for row in pred_rows:
        label = f"P:{row['label_name']} {row['score']:.2f}"
        draw_box(draw, scale_box(row_box(row), scale), pred_color(row), label, font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def render_crop(
    image_path: str,
    gt_rows: List[Dict],
    pred_rows: List[Dict],
    focus_box: Tuple[float, float, float, float],
    out_path: Path,
    min_crop: int = 220,
    margin: float = 3.0,
) -> None:
    image = Image.open(image_path).convert("RGB")
    crop = expanded_crop_box(focus_box, image.size, min_crop=min_crop, margin=margin)
    cropped = image.crop(crop)
    draw = ImageDraw.Draw(cropped)
    font = ImageFont.load_default()

    for row in gt_rows:
        box = intersect_box_with_crop(row_box(row), crop)
        if box is not None:
            draw_box(draw, box, gt_color(row), f"GT:{row['label_name']}", font)
    for row in pred_rows:
        box = intersect_box_with_crop(row_box(row), crop)
        if box is not None:
            draw_box(draw, box, pred_color(row), f"P:{row['label_name']} {row['score']:.2f}", font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path)


def resize_for_display(image: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = image.resize((int(width * scale), int(height * scale)))
    return resized, scale


def scale_box(box: Tuple[float, float, float, float], scale: float) -> Tuple[float, float, float, float]:
    return tuple(value * scale for value in box)


def row_box(row: Dict) -> Tuple[float, float, float, float]:
    return (row["x1"], row["y1"], row["x2"], row["y2"])


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    color: str,
    label: str,
    font,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    if label:
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_h - 4)
        draw.rectangle([x1, text_y, x1 + text_w + 4, text_y + text_h + 4], fill=color)
        draw.text((x1 + 2, text_y + 2), label, fill="white", font=font)


def gt_color(row: Dict) -> str:
    if row["status"] == "tp":
        return "#2f855a"
    if row["reason"] == "misclassification":
        return "#dd6b20"
    return "#d69e2e"


def pred_color(row: Dict) -> str:
    if row["status"] == "tp":
        return "#2b6cb0"
    if row["reason"] == "misclassification":
        return "#805ad5"
    return "#c53030"


def expanded_crop_box(
    box: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    min_crop: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(x2 - x1, 1.0) * margin
    height = max(y2 - y1, 1.0) * margin
    crop_w = max(min_crop, int(math.ceil(width)))
    crop_h = max(min_crop, int(math.ceil(height)))
    img_w, img_h = image_size

    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))
    left = max(0, min(left, img_w - crop_w))
    top = max(0, min(top, img_h - crop_h))
    right = min(img_w, left + crop_w)
    bottom = min(img_h, top + crop_h)
    return left, top, right, bottom


def intersect_box_with_crop(
    box: Tuple[float, float, float, float],
    crop: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float, float, float]]:
    left, top, right, bottom = crop
    x1, y1, x2, y2 = box
    if x2 < left or x1 > right or y2 < top or y1 > bottom:
        return None
    return (x1 - left, y1 - top, x2 - left, y2 - top)


class ActivationRecorder:
    def __init__(self, module: torch.nn.Module) -> None:
        self.activations = None
        self.gradients = None
        self.forward_handle = module.register_forward_hook(self._forward_hook)
        self.backward_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output[0] if isinstance(output, (tuple, list)) else output

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        grad = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        self.gradients = grad

    def compute_cam(self) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        activations = self.activations
        gradients = self.gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.detach().cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


class DecoderInputRecorder:
    def __init__(self, module: torch.nn.Module) -> None:
        self.query_embeddings = None
        self.forward_handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        tensor_input = next((item for item in inputs if torch.is_tensor(item)), None)
        self.query_embeddings = tensor_input.detach().float().cpu() if tensor_input is not None else None

    def pop(self) -> Optional[torch.Tensor]:
        query_embeddings = self.query_embeddings
        self.query_embeddings = None
        return query_embeddings

    def close(self) -> None:
        self.forward_handle.remove()


def save_gradcam_examples(
    analysis_dir: Path,
    model: torch.nn.Module,
    dataset,
    pred_records: List[Dict],
    per_image_rows: List[Dict],
    device: torch.device,
    cam_module_name: str,
    num_gradcam: int,
) -> None:
    selected = select_gradcam_predictions(pred_records, per_image_rows, num_gradcam)
    if not selected:
        return

    cam_module = resolve_cam_module(model, cam_module_name)
    out_dir = analysis_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    for rank, row in enumerate(selected):
        with torch.enable_grad():
            try:
                sample, target = dataset[row["dataset_idx"]]
                sample = sample.unsqueeze(0).to(device)
                recorder = ActivationRecorder(cam_module)
                model.zero_grad(set_to_none=True)
                outputs = model(sample)
                score = outputs["pred_logits"][0, int(row["query_idx"]), int(row["class_idx"])]
                score.backward()
                cam = recorder.compute_cam()
                recorder.close()

                image = tensor_to_pil(sample[0].detach().cpu())
                overlay = overlay_cam_on_image(image, cam, alpha=0.45)

                orig_w, orig_h = int(target["orig_size"][0].item()), int(target["orig_size"][1].item())
                input_h, input_w = sample.shape[-2], sample.shape[-1]
                pred_box = scale_prediction_box_to_input(row_box(row), orig_w, orig_h, input_w, input_h)
                draw = ImageDraw.Draw(overlay)
                font = ImageFont.load_default()
                label = f"{row['status'].upper()} {row['label_name']} {row['score']:.2f}"
                draw_box(draw, pred_box, "#ffffff", label, font)
                overlay.save(out_dir / f"{rank:02d}_{row['status']}_{row['reason']}_img{row['image_id']}.png")
            except Exception as exc:
                failures.append(
                    {
                        "image_id": row["image_id"],
                        "pred_index": row["pred_index"],
                        "reason": row["reason"],
                        "error": str(exc),
                    }
                )
                print(
                    f"[warn] gradcam failed for image_id={row['image_id']}, "
                    f"pred_index={row['pred_index']}: {exc}"
                )

    if failures:
        write_json(out_dir / "gradcam_failures.json", {"failures": failures})


def select_gradcam_predictions(
    pred_records: List[Dict],
    per_image_rows: List[Dict],
    limit: int,
) -> List[Dict]:
    image_lookup = {row["image_id"]: row for row in per_image_rows}
    selected = []
    used = set()

    def add_first(candidates: List[Dict]) -> None:
        for candidate in candidates:
            key = (candidate["image_id"], candidate["pred_index"])
            if key not in used:
                selected.append(candidate)
                used.add(key)
                break

    tps = sorted([row for row in pred_records if row["status"] == "tp"], key=lambda row: -row["score"])
    fps = sorted([row for row in pred_records if row["status"] == "fp"], key=lambda row: -row["score"])
    tiny_tps = [
        row for row in tps if row.get("matched_gt_size_bin") in {"very_tiny", "tiny"}
    ]
    crowded_fps = [
        row for row in fps if image_lookup.get(row["image_id"], {}).get("num_gt", 0) >= 6
    ]

    add_first(tps)
    add_first(fps)
    add_first(tiny_tps)
    add_first(crowded_fps)

    for candidate in tps + fps:
        if len(selected) >= limit:
            break
        key = (candidate["image_id"], candidate["pred_index"])
        if key in used:
            continue
        selected.append(candidate)
        used.add(key)
    return selected[:limit]


def build_tsne_samples(pred_rows: List[Dict], query_embeddings: torch.Tensor) -> List[Dict]:
    if not pred_rows or query_embeddings is None:
        return []
    if not torch.is_tensor(query_embeddings):
        query_embeddings = torch.as_tensor(query_embeddings)
    if query_embeddings.ndim != 2:
        return []

    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for row in pred_rows:
        if "query_idx" not in row:
            continue
        grouped[int(row["query_idx"])].append(row)

    samples: List[Dict] = []
    for query_idx, rows in grouped.items():
        if query_idx < 0 or query_idx >= query_embeddings.shape[0]:
            continue
        best_row = max(
            rows,
            key=lambda row: (
                float(row.get("score", 0.0)),
                row.get("status") == "tp",
                -int(row.get("pred_index", 0)),
            ),
        )
        embedding = query_embeddings[query_idx].detach().float().cpu().numpy().copy()
        samples.append(
            {
                "query_uid": f"{best_row['image_id']}:{query_idx}",
                "image_id": int(best_row["image_id"]),
                "dataset_idx": int(best_row.get("dataset_idx", -1)),
                "pred_index": int(best_row["pred_index"]),
                "query_idx": query_idx,
                "label_id": int(best_row["label_id"]),
                "label_name": str(best_row["label_name"]),
                "status": str(best_row["status"]),
                "reason": str(best_row["reason"]),
                "score": float(best_row["score"]),
                "matched_iou": float(best_row.get("matched_iou", best_row.get("best_any_iou", 0.0))),
                "pred_size_bin": str(best_row.get("pred_size_bin", "")),
                "matched_gt_size_bin": str(best_row.get("matched_gt_size_bin", "")),
                "scene_bin": str(best_row.get("scene_bin", "")),
                "num_gt": int(best_row.get("num_gt", 0)),
                "pair_overlap_count": int(best_row.get("pair_overlap_count", 0)),
                "embedding": embedding,
            }
        )
    return samples


def save_tsne_outputs(
    analysis_dir: Path,
    tsne_samples: List[Dict],
    class_ids: Sequence[int],
    category_names: Dict[int, str],
    args: argparse.Namespace,
) -> None:
    tsne_dir = analysis_dir / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)

    if not tsne_samples:
        write_json(
            tsne_dir / "tsne_summary.json",
            {
                "num_input_samples": 0,
                "num_sampled_points": 0,
                "notes": ["No prediction queries were collected for t-SNE."],
            },
        )
        print("[tsne] no samples collected; skipping t-SNE")
        return

    sampled_samples = sample_tsne_samples(
        tsne_samples,
        max_points=max(2, int(args.tsne_max_points)),
        random_state=int(args.tsne_random_state),
    )
    if len(sampled_samples) < 2:
        write_json(
            tsne_dir / "tsne_summary.json",
            {
                "num_input_samples": len(tsne_samples),
                "num_sampled_points": len(sampled_samples),
                "notes": ["Not enough samples for t-SNE."],
            },
        )
        print("[tsne] not enough samples for t-SNE")
        return

    features = np.stack([sample["embedding"] for sample in sampled_samples], axis=0).astype(np.float64)
    coords = compute_tsne_embedding(
        features=features,
        perplexity=float(args.tsne_perplexity),
        n_iter=int(args.tsne_iterations),
        learning_rate=float(args.tsne_learning_rate),
        pca_dim=int(args.tsne_pca_dim),
        random_state=int(args.tsne_random_state),
    )

    rows = []
    for sample, coord in zip(sampled_samples, coords):
        rows.append(
            {
                "tsne_x": float(coord[0]),
                "tsne_y": float(coord[1]),
                "query_uid": sample["query_uid"],
                "image_id": sample["image_id"],
                "dataset_idx": sample["dataset_idx"],
                "pred_index": sample["pred_index"],
                "query_idx": sample["query_idx"],
                "label_id": sample["label_id"],
                "label_name": sample["label_name"],
                "status": sample["status"],
                "reason": sample["reason"],
                "score": sample["score"],
                "matched_iou": sample["matched_iou"],
                "pred_size_bin": sample["pred_size_bin"],
                "matched_gt_size_bin": sample["matched_gt_size_bin"],
                "scene_bin": sample["scene_bin"],
                "num_gt": sample["num_gt"],
                "pair_overlap_count": sample["pair_overlap_count"],
            }
        )

    write_csv(tsne_dir / "tsne_embeddings.csv", rows)

    status_counts = Counter(sample["status"] for sample in sampled_samples)
    reason_counts = Counter(sample["reason"] for sample in sampled_samples)
    class_counts = Counter(sample["label_name"] for sample in sampled_samples)
    size_bins = [
        sample["matched_gt_size_bin"] if sample["matched_gt_size_bin"] else sample["pred_size_bin"]
        for sample in sampled_samples
    ]
    size_counts = Counter(size_bins)

    write_json(
        tsne_dir / "tsne_summary.json",
        {
            "num_input_samples": len(tsne_samples),
            "num_sampled_points": len(sampled_samples),
            "max_points": int(args.tsne_max_points),
            "status_counts": dict(status_counts),
            "reason_counts": dict(reason_counts),
            "class_counts": dict(class_counts),
            "size_bin_counts": dict(size_counts),
            "embedding_source": "decoder_query_embeddings",
            "selection_policy": "status-balanced, label-aware subsampling with one point per kept query",
            "perplexity": float(args.tsne_perplexity),
            "iterations": int(args.tsne_iterations),
            "learning_rate": float(args.tsne_learning_rate),
            "pca_dim": int(args.tsne_pca_dim),
            "notes": [
                "t-SNE embeddings are built from decoder query embeddings",
                "One point per kept query is used after deduplicating class projections from focal-loss style top-k outputs",
            ],
        },
    )

    status_labels = [sample["status"] for sample in sampled_samples]
    reason_labels = [sample["reason"] for sample in sampled_samples]
    class_labels = [sample["label_name"] for sample in sampled_samples]
    size_labels = size_bins
    confidence_values = [sample["score"] for sample in sampled_samples]

    plot_tsne_categorical(
        points=coords,
        labels=status_labels,
        out_path=tsne_dir / "tsne_by_status.png",
        title="t-SNE by status",
        order=["tp", "fp"],
        color_map={"tp": "#2f855a", "fp": "#c53030"},
    )
    plot_tsne_categorical(
        points=coords,
        labels=class_labels,
        out_path=tsne_dir / "tsne_by_class.png",
        title="t-SNE by class",
        order=[category_names.get(class_id, str(class_id)) for class_id in class_ids],
        palette="tab10",
    )
    plot_tsne_categorical(
        points=coords,
        labels=reason_labels,
        out_path=tsne_dir / "tsne_by_reason.png",
        title="t-SNE by reason",
        order=sorted(set(reason_labels), key=lambda name: (name != "tp", name)),
        palette="tab20",
    )
    plot_tsne_categorical(
        points=coords,
        labels=size_labels,
        out_path=tsne_dir / "tsne_by_size.png",
        title="t-SNE by size bin",
        order=SIZE_BIN_ORDER,
        color_map={
            "very_tiny": "#2b6cb0",
            "tiny": "#3182ce",
            "small": "#38a169",
            "medium": "#d69e2e",
            "large": "#c53030",
            "": "#718096",
        },
    )
    plot_tsne_continuous(
        points=coords,
        values=confidence_values,
        out_path=tsne_dir / "tsne_by_confidence.png",
        title="t-SNE by confidence",
        colorbar_label="Confidence",
    )

    print(f"[tsne] saved {len(sampled_samples)} points to {tsne_dir}")


def sample_tsne_samples(
    tsne_samples: List[Dict],
    max_points: int,
    random_state: int,
) -> List[Dict]:
    if len(tsne_samples) <= max_points:
        return list(tsne_samples)

    rng = random.Random(random_state)
    by_status: Dict[str, List[Dict]] = defaultdict(list)
    for sample in tsne_samples:
        by_status[str(sample["status"])].append(sample)

    selected_ids = set()
    selected: List[Dict] = []

    if "tp" in by_status and "fp" in by_status:
        tp_budget = max_points // 2
        fp_budget = max_points - tp_budget
        selected.extend(select_balanced_subset(by_status["tp"], tp_budget, rng))
        selected.extend(select_balanced_subset(by_status["fp"], fp_budget, rng))
    else:
        for status_samples in by_status.values():
            selected.extend(select_balanced_subset(status_samples, max_points, rng))

    selected = dedupe_samples_by_uid(selected)
    for sample in selected:
        selected_ids.add(sample["query_uid"])

    if len(selected) < max_points:
        remainder = [sample for sample in tsne_samples if sample["query_uid"] not in selected_ids]
        if remainder:
            extra = rng.sample(remainder, min(max_points - len(selected), len(remainder)))
            selected.extend(extra)

    return dedupe_samples_by_uid(selected)[:max_points]


def select_balanced_subset(samples: List[Dict], budget: int, rng: random.Random) -> List[Dict]:
    if budget <= 0 or not samples:
        return []

    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for sample in samples:
        grouped[int(sample["label_id"])].append(sample)

    selected: List[Dict] = []
    selected_ids = set()
    label_order = sorted(grouped.keys(), key=lambda label: len(grouped[label]), reverse=True)

    if budget >= len(label_order):
        for label in label_order:
            choice = rng.choice(grouped[label])
            selected.append(choice)
            selected_ids.add(choice["query_uid"])
        remaining_budget = budget - len(label_order)
        if remaining_budget > 0:
            remainder = [sample for sample in samples if sample["query_uid"] not in selected_ids]
            if remainder:
                extra = rng.sample(remainder, min(remaining_budget, len(remainder)))
                selected.extend(extra)
    else:
        for label in label_order[:budget]:
            choice = rng.choice(grouped[label])
            selected.append(choice)

    return selected


def dedupe_samples_by_uid(samples: List[Dict]) -> List[Dict]:
    deduped = []
    seen = set()
    for sample in samples:
        uid = sample["query_uid"]
        if uid in seen:
            continue
        deduped.append(sample)
        seen.add(uid)
    return deduped


def compute_tsne_embedding(
    features: np.ndarray,
    perplexity: float,
    n_iter: int,
    learning_rate: float,
    pca_dim: int,
    random_state: int,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    n_samples = features.shape[0]
    if n_samples <= 2:
        return np.zeros((n_samples, 2), dtype=np.float64)

    features = features - features.mean(axis=0, keepdims=True)
    features = pca_reduce(features, pca_dim)
    perplexity = float(max(1.0, min(perplexity, max(1.0, (n_samples - 1) / 3.0))))
    P = compute_joint_probabilities(features, perplexity)

    rng = np.random.default_rng(random_state)
    if features.shape[1] >= 2:
        y = features[:, :2].copy()
        y = y / (np.std(y, axis=0, keepdims=True) + 1e-12)
        y = y * 1e-4 + rng.standard_normal(size=(n_samples, 2)) * 1e-4
    else:
        y = rng.standard_normal(size=(n_samples, 2)) * 1e-4

    gains = np.ones_like(y)
    y_inc = np.zeros_like(y)
    exaggeration_iters = min(250, max(50, n_iter // 4))
    early_exaggeration = 12.0

    print(
        f"[tsne] optimizing {n_samples} points "
        f"(perplexity={perplexity:.1f}, iterations={n_iter}, pca_dim={pca_dim})"
    )
    for it in range(max(1, n_iter)):
        dist_sq = pairwise_squared_distances(y)
        num = 1.0 / (1.0 + dist_sq)
        np.fill_diagonal(num, 0.0)
        sum_num = np.sum(num)
        if not np.isfinite(sum_num) or sum_num <= 0:
            sum_num = 1e-12
        Q = num / sum_num
        P_use = P * early_exaggeration if it < exaggeration_iters else P
        forces = (P_use - Q) * num
        grad = 4.0 * (np.sum(forces, axis=1, keepdims=True) * y - forces @ y)

        momentum = 0.5 if it < 250 else 0.8
        gains = np.where(np.sign(grad) != np.sign(y_inc), gains + 0.2, gains * 0.8)
        gains = np.clip(gains, 0.01, None)
        y_inc = momentum * y_inc - learning_rate * gains * grad
        y += y_inc
        y -= y.mean(axis=0, keepdims=True)

        if (it + 1) % 250 == 0 or it + 1 == n_iter:
            kl = float(np.sum(P * np.log((P + 1e-12) / (Q + 1e-12))))
            print(f"[tsne] iter {it + 1}/{n_iter} kl={kl:.4f}")

    return y


def pca_reduce(features: np.ndarray, pca_dim: int) -> np.ndarray:
    if pca_dim <= 0 or features.shape[1] <= pca_dim:
        return features
    pca_dim = min(pca_dim, features.shape[0] - 1, features.shape[1])
    if pca_dim <= 0:
        return features
    u, s, _ = np.linalg.svd(features, full_matrices=False)
    return u[:, :pca_dim] * s[:pca_dim]


def compute_joint_probabilities(features: np.ndarray, perplexity: float) -> np.ndarray:
    dist_sq = pairwise_squared_distances(features)
    n_samples = dist_sq.shape[0]
    probabilities = np.zeros((n_samples, n_samples), dtype=np.float64)
    target_entropy = math.log(perplexity)

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        row = dist_sq[i, mask]
        h, this_p = entropy_and_probabilities(row, beta=1.0)
        h_diff = h - target_entropy
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf
        tries = 0
        while abs(h_diff) > 1e-5 and tries < 50:
            if h_diff > 0:
                beta_min = beta
                beta = beta * 2.0 if np.isinf(beta_max) else 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = beta / 2.0 if np.isinf(beta_min) else 0.5 * (beta + beta_min)
            h, this_p = entropy_and_probabilities(row, beta=beta)
            h_diff = h - target_entropy
            tries += 1
        probabilities[i, mask] = this_p

    probabilities = probabilities + probabilities.T
    probabilities = np.maximum(probabilities, 1e-12)
    probabilities /= np.sum(probabilities)
    return probabilities


def entropy_and_probabilities(distances: np.ndarray, beta: float) -> Tuple[float, np.ndarray]:
    p = np.exp(-distances * beta)
    total = np.sum(p)
    if not np.isfinite(total) or total <= 0:
        p = np.full_like(distances, 1.0 / max(len(distances), 1))
        return math.log(max(len(distances), 1)), p
    p = p / total
    entropy = math.log(total) + beta * float(np.sum(distances * p))
    return entropy, p


def pairwise_squared_distances(features: np.ndarray) -> np.ndarray:
    sum_sq = np.sum(features * features, axis=1, keepdims=True)
    dist_sq = sum_sq + sum_sq.T - 2.0 * features @ features.T
    dist_sq = np.maximum(dist_sq, 0.0)
    np.fill_diagonal(dist_sq, 0.0)
    return dist_sq


def plot_tsne_categorical(
    points: np.ndarray,
    labels: Sequence[str],
    out_path: Path,
    title: str,
    order: Optional[Sequence[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    palette: str = "tab10",
) -> None:
    if len(points) == 0:
        return
    labels = np.asarray([str(label) for label in labels], dtype=object)
    if order is None:
        order = list(dict.fromkeys(labels.tolist()))
    order = [str(label) for label in order if np.any(labels == str(label))]
    if not order:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cmap = None if color_map is not None else plt.get_cmap(palette, len(order))
    for idx, label in enumerate(order):
        mask = labels == label
        if not np.any(mask):
            continue
        color = color_map.get(label) if color_map is not None else cmap(idx)
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=12,
            alpha=0.8,
            linewidths=0,
            color=color,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tsne_continuous(
    points: np.ndarray,
    values: Sequence[float],
    out_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    if len(points) == 0:
        return
    values = np.asarray(values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=values,
        cmap=cmap,
        s=12,
        alpha=0.8,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def resolve_cam_module(model: torch.nn.Module, cam_module_name: str) -> torch.nn.Module:
    backbone = model.backbone
    if cam_module_name == "backbone_last_stage":
        return backbone.stages[-1]
    if cam_module_name == "backbone_last_return":
        stage_idx = max(backbone.return_idx) if hasattr(backbone, "return_idx") else -1
        return backbone.stages[stage_idx]
    raise ValueError(cam_module_name)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().clamp(0, 1).cpu()
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def overlay_cam_on_image(image: Image.Image, cam: np.ndarray, alpha: float) -> Image.Image:
    cam_image = Image.fromarray(np.uint8(plt.get_cmap("turbo")(cam)[:, :, :3] * 255)).resize(image.size)
    image_np = np.asarray(image).astype(np.float32)
    cam_np = np.asarray(cam_image).astype(np.float32)
    blended = np.clip((1.0 - alpha) * image_np + alpha * cam_np, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def scale_prediction_box_to_input(
    box: Tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    input_w: int,
    input_h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        x1 * input_w / max(orig_w, 1),
        y1 * input_h / max(orig_h, 1),
        x2 * input_w / max(orig_w, 1),
        y2 * input_h / max(orig_h, 1),
    )


if __name__ == "__main__":
    main()
