"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(
        self,
        weight_dict,
        use_focal_loss=False,
        alpha=0.25,
        gamma=2.0,
        scale_adaptive_enabled=False,
        tiny_area_lower=8**2,
        tiny_area_upper=16**2,
        tiny_cost_class=None,
        tiny_cost_bbox=None,
        tiny_cost_giou=None,
        tiny_cost_center=0.0,
        tiny_cost_nwd=0.0,
        tiny_cost_uncertainty=0.0,
        uncertainty_type="entropy",
        eps=1e-8,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.scale_adaptive_enabled = scale_adaptive_enabled
        self.tiny_area_lower = float(tiny_area_lower)
        self.tiny_area_upper = float(tiny_area_upper)
        self.tiny_cost_class = self.cost_class if tiny_cost_class is None else tiny_cost_class
        self.tiny_cost_bbox = self.cost_bbox if tiny_cost_bbox is None else tiny_cost_bbox
        self.tiny_cost_giou = self.cost_giou if tiny_cost_giou is None else tiny_cost_giou
        self.tiny_cost_center = tiny_cost_center
        self.tiny_cost_nwd = tiny_cost_nwd
        self.tiny_cost_uncertainty = tiny_cost_uncertainty
        self.uncertainty_type = uncertainty_type
        self.eps = eps

        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0
        ), "all costs cant be 0"
        assert self.tiny_area_lower < self.tiny_area_upper, "tiny area range must be valid"
        assert self.uncertainty_type in ("entropy", "variance"), "unsupported uncertainty type"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"])
        else:
            out_prob = outputs["pred_logits"].softmax(-1)

        out_bbox = outputs["pred_boxes"]
        out_corners = outputs.get("pred_corners")

        indices_pre = []
        cost_matrices = []
        for batch_idx in range(bs):
            tgt_ids = targets[batch_idx]["labels"]
            tgt_bbox = targets[batch_idx]["boxes"]

            if tgt_ids.numel() == 0:
                indices_pre.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
                cost_matrices.append(torch.zeros((num_queries, 0), dtype=torch.float32))
                continue

            batch_prob = out_prob[batch_idx]
            batch_bbox = out_bbox[batch_idx]
            batch_corners = out_corners[batch_idx] if out_corners is not None else None

            # Contrary to the loss, we don't use the NLL, but approximate it in 1 - p[class].
            if self.use_focal_loss:
                batch_prob = batch_prob[:, tgt_ids]
                neg_cost_class = (
                    (1 - self.alpha)
                    * (batch_prob**self.gamma)
                    * (-(1 - batch_prob + self.eps).log())
                )
                pos_cost_class = (
                    self.alpha
                    * ((1 - batch_prob) ** self.gamma)
                    * (-(batch_prob + self.eps).log())
                )
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -batch_prob[:, tgt_ids]

            cost_bbox = torch.cdist(batch_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(batch_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )

            cost = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )

            if self.scale_adaptive_enabled:
                cost = self._apply_scale_adaptive_cost(
                    cost,
                    cost_class,
                    cost_bbox,
                    cost_giou,
                    batch_bbox,
                    tgt_bbox,
                    batch_corners,
                    targets[batch_idx],
                )

            cost = torch.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
            cost_cpu = cost.cpu()
            cost_matrices.append(cost_cpu)
            indices_pre.append(linear_sum_assignment(cost_cpu.numpy()))

        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {"indices_o2m": self.get_top_k_matches(cost_matrices, k=return_topk, initial_indices=indices_pre)}

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def _apply_scale_adaptive_cost(
        self,
        cost,
        cost_class,
        cost_bbox,
        cost_giou,
        pred_boxes,
        tgt_boxes,
        pred_corners,
        target,
    ):
        pred_boxes_abs, tgt_boxes_abs, target_areas = self._to_absolute_boxes(
            pred_boxes, tgt_boxes, target, self._get_target_areas(target, tgt_boxes)
        )
        tiny_mask = (target_areas >= self.tiny_area_lower) & (target_areas < self.tiny_area_upper)
        if not tiny_mask.any():
            return cost

        tiny_cost = (
            self.tiny_cost_bbox * cost_bbox
            + self.tiny_cost_class * cost_class
            + self.tiny_cost_giou * cost_giou
        )

        if self.tiny_cost_center > 0:
            tiny_cost = tiny_cost + self.tiny_cost_center * self._compute_center_cost(
                pred_boxes_abs, tgt_boxes_abs, target_areas
            )

        if self.tiny_cost_nwd > 0:
            tiny_cost = tiny_cost + self.tiny_cost_nwd * self._compute_nwd_cost(
                pred_boxes_abs, tgt_boxes_abs, target_areas
            )

        if self.tiny_cost_uncertainty > 0 and pred_corners is not None:
            uncertainty = self._compute_query_uncertainty(pred_corners).unsqueeze(-1)
            tiny_cost = tiny_cost + self.tiny_cost_uncertainty * uncertainty

        adaptive_cost = cost.clone()
        adaptive_cost[:, tiny_mask] = tiny_cost[:, tiny_mask]
        return adaptive_cost

    def _to_absolute_boxes(self, pred_boxes, tgt_boxes, target, target_areas):
        image_size = self._get_image_size(target, pred_boxes.device, pred_boxes.dtype)
        scale = torch.cat((image_size, image_size), dim=0)
        pred_boxes_abs = pred_boxes * scale
        tgt_boxes_abs = tgt_boxes * scale
        if target_areas is None:
            target_areas = tgt_boxes_abs[:, 2] * tgt_boxes_abs[:, 3]
        return pred_boxes_abs, tgt_boxes_abs, target_areas

    def _get_target_areas(self, target, tgt_boxes):
        if "area" in target and len(target["area"]) == len(tgt_boxes):
            return target["area"].to(device=tgt_boxes.device, dtype=tgt_boxes.dtype)
        return None

    def _get_image_size(self, target, device, dtype):
        if "size" in target:
            size = target["size"].to(device=device, dtype=dtype)
            return torch.stack((size[1], size[0]))
        if "orig_size" in target:
            return target["orig_size"].to(device=device, dtype=dtype)
        return torch.ones(2, device=device, dtype=dtype)

    def _compute_center_cost(self, pred_boxes_abs, tgt_boxes_abs, target_areas):
        pred_center = pred_boxes_abs[:, :2]
        tgt_center = tgt_boxes_abs[:, :2]
        center_delta = pred_center[:, None, :] - tgt_center[None, :, :]
        center_dist = torch.linalg.vector_norm(center_delta, dim=-1)
        normalizer = target_areas.clamp(min=1.0).sqrt().unsqueeze(0)
        return center_dist / normalizer

    def _compute_nwd_cost(self, pred_boxes_abs, tgt_boxes_abs, target_areas):
        center_sq = (pred_boxes_abs[:, None, :2] - tgt_boxes_abs[None, :, :2]).pow(2).sum(-1)
        shape_sq = 0.25 * (
            pred_boxes_abs[:, None, 2:] - tgt_boxes_abs[None, :, 2:]
        ).pow(2).sum(-1)
        wasserstein = torch.sqrt((center_sq + shape_sq).clamp(min=0.0))
        normalizer = target_areas.clamp(min=1.0).sqrt().unsqueeze(0)
        return 1 - torch.exp(-wasserstein / normalizer.clamp(min=self.eps))

    def _compute_query_uncertainty(self, pred_corners):
        num_bins = pred_corners.shape[-1] // 4
        if pred_corners.shape[-1] % 4 != 0 or num_bins <= 1:
            return pred_corners.new_zeros(pred_corners.shape[0])

        corner_prob = F.softmax(pred_corners.reshape(pred_corners.shape[0], 4, num_bins), dim=-1)
        if self.uncertainty_type == "entropy":
            uncertainty = -(corner_prob * torch.log(corner_prob.clamp(min=self.eps))).sum(-1)
            return uncertainty.mean(-1) / max(math.log(num_bins), self.eps)

        bins = torch.arange(num_bins, device=pred_corners.device, dtype=pred_corners.dtype)
        bins = bins.view(1, 1, num_bins)
        mean = (corner_prob * bins).sum(-1, keepdim=True)
        variance = (corner_prob * (bins - mean).pow(2)).sum(-1).mean(-1)
        return variance / max((num_bins - 1) ** 2, 1)

    def get_top_k_matches(self, cost_matrices, k=1, initial_indices=None):
        all_matches = [[] for _ in cost_matrices]
        working_costs = [cost.clone() for cost in cost_matrices]

        for topk_idx in range(k):
            if topk_idx == 0 and initial_indices is not None:
                indices_k = initial_indices
            else:
                indices_k = []
                for cost in working_costs:
                    if cost.shape[1] == 0:
                        indices_k.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
                    else:
                        indices_k.append(linear_sum_assignment(cost.numpy()))

            for batch_idx, (src_idx, tgt_idx) in enumerate(indices_k):
                src_idx = torch.as_tensor(src_idx, dtype=torch.int64)
                tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.int64)
                all_matches[batch_idx].append((src_idx, tgt_idx))

                if src_idx.numel() > 0:
                    working_costs[batch_idx][src_idx, tgt_idx] = 1e6

        merged_matches = []
        for match_group in all_matches:
            if not match_group:
                merged_matches.append(
                    (
                        torch.zeros(0, dtype=torch.int64),
                        torch.zeros(0, dtype=torch.int64),
                    )
                )
                continue
            merged_matches.append(
                (
                    torch.cat([match[0] for match in match_group], dim=0),
                    torch.cat([match[1] for match in match_group], dim=0),
                )
            )
        return merged_matches
