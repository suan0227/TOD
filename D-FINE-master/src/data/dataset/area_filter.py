"""
Utilities for filtering COCO-style annotations by object area.
"""

import copy
from typing import Iterable, List, Optional, Sequence, Tuple

DEFAULT_AREA_BUCKETS = {
    "all": (0.0, float("inf")),
    "very_tiny": (float(2**2), float(8**2)),
    "tiny": (float(8**2), float(16**2)),
    "small": (float(16**2), float(32**2)),
    "medium": (float(32**2), float(96**2)),
    "large": (float(96**2), float("inf")),
}

AreaRange = Tuple[float, float]


def resolve_area_ranges(
    area_labels: Optional[Sequence[str]] = None,
    area_ranges: Optional[Iterable[Sequence[float]]] = None,
) -> List[AreaRange]:
    resolved_ranges: List[AreaRange] = []

    if area_labels:
        unknown_labels = [label for label in area_labels if label not in DEFAULT_AREA_BUCKETS]
        if unknown_labels:
            raise ValueError(
                f"Unknown area labels: {unknown_labels}. "
                f"Available labels: {sorted(DEFAULT_AREA_BUCKETS.keys())}"
            )

        resolved_ranges.extend(DEFAULT_AREA_BUCKETS[label] for label in area_labels)

    if area_ranges:
        for area_range in area_ranges:
            if len(area_range) != 2:
                raise ValueError(f"Each area range must contain exactly 2 values, got {area_range}")

            min_area, max_area = float(area_range[0]), float(area_range[1])
            if min_area > max_area:
                raise ValueError(
                    f"Each area range must satisfy min_area <= max_area, got {area_range}"
                )
            resolved_ranges.append((min_area, max_area))

    return merge_overlapping_area_ranges(resolved_ranges)


def merge_overlapping_area_ranges(area_ranges: Iterable[AreaRange]) -> List[AreaRange]:
    sorted_ranges = sorted(area_ranges, key=lambda value: (value[0], value[1]))
    if not sorted_ranges:
        return []

    merged_ranges: List[AreaRange] = [sorted_ranges[0]]
    for min_area, max_area in sorted_ranges[1:]:
        prev_min, prev_max = merged_ranges[-1]
        if min_area <= prev_max:
            merged_ranges[-1] = (prev_min, max(prev_max, max_area))
        else:
            merged_ranges.append((min_area, max_area))
    return merged_ranges


def get_annotation_area(annotation: dict) -> float:
    if annotation.get("area") is not None:
        return float(annotation["area"])

    bbox = annotation.get("bbox")
    if bbox is None or len(bbox) < 4:
        return 0.0

    width = max(float(bbox[2]), 0.0)
    height = max(float(bbox[3]), 0.0)
    return width * height


def is_annotation_in_area_ranges(annotation: dict, area_ranges: Sequence[AreaRange]) -> bool:
    if not area_ranges:
        return True

    area = get_annotation_area(annotation)
    return any(min_area <= area < max_area for min_area, max_area in area_ranges)


def filter_annotations_by_area(annotations: List[dict], area_ranges: Sequence[AreaRange]) -> List[dict]:
    if not area_ranges:
        return annotations

    return [annotation for annotation in annotations if is_annotation_in_area_ranges(annotation, area_ranges)]


def filter_coco_dataset_dict(
    coco_dataset: dict,
    area_ranges: Sequence[AreaRange],
    exclude_images_without_annotations: bool = False,
) -> dict:
    filtered_dataset = copy.deepcopy(coco_dataset)
    filtered_annotations = filter_annotations_by_area(
        filtered_dataset.get("annotations", []),
        area_ranges,
    )
    filtered_dataset["annotations"] = filtered_annotations

    if exclude_images_without_annotations:
        kept_image_ids = {annotation["image_id"] for annotation in filtered_annotations}
        filtered_dataset["images"] = [
            image for image in filtered_dataset.get("images", []) if image["id"] in kept_image_ids
        ]

    return filtered_dataset
