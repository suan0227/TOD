"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

# MiXaiLL76 replacing pycocotools with faster-coco-eval for better performance and support.
"""

from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

from ...core import register

__all__ = [
    "CocoEvaluator",
]


@register()
class CocoEvaluator(FasterCocoEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_custom_area_bins()

    def _apply_custom_area_bins(self):
        # Align area bins with AITOD tiny-object analysis used in RT-DETR.
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.areaRng = [
                [0**2, 1e5**2],  # all
                [2**2, 8**2],  # very_tiny
                [8**2, 16**2],  # tiny
                [16**2, 32**2],  # small
                [32**2, 96**2],  # medium
                [96**2, 1e5**2],  # large
            ]
            coco_eval.params.areaRngLbl = [
                "all",
                "very_tiny",
                "tiny",
                "small",
                "medium",
                "large",
            ]

    def cleanup(self):
        super().cleanup()
        # `cleanup` may reset evaluator params; re-apply custom bins.
        self._apply_custom_area_bins()
