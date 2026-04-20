from __future__ import annotations

import torch
import torch.nn as nn

from ..box_ops import box_cxcywh_to_xyxy, box_iou
from .contrastive_loss import contrastive_repulsion_loss
from .prototype_bank import PrototypeBank


class ContrastiveHead(nn.Module):
    def __init__(
        self,
        bank_size: int = 64,
        momentum: float = 0.9,
        margin: float = 0.4,
        bg_iou_threshold: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        if bg_iou_threshold < 0.0:
            raise ValueError("bg_iou_threshold must be non-negative")

        self.margin = float(margin)
        self.bg_iou_threshold = float(bg_iou_threshold)
        self.prototype_bank = PrototypeBank(bank_size=bank_size, momentum=momentum, eps=eps)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets,
        indices,
    ) -> torch.Tensor:
        foreground_embeddings, background_embeddings = self._collect_embeddings(
            query_embeddings, pred_boxes, targets, indices
        )

        prototypes = self.prototype_bank.active_prototypes()
        if foreground_embeddings.numel() == 0 or prototypes.numel() == 0:
            loss = query_embeddings.sum() * 0.0
        else:
            loss = contrastive_repulsion_loss(
                foreground_embeddings, prototypes, margin=self.margin
            )

        if self.training:
            self.prototype_bank.update(background_embeddings)
        return loss

    def _collect_embeddings(self, query_embeddings, pred_boxes, targets, indices):
        fg_embeddings = []
        bg_embeddings = []

        for batch_queries, batch_boxes, target, match in zip(
            query_embeddings, pred_boxes, targets, indices
        ):
            matched_src = match[0]
            matched_mask = torch.zeros(
                batch_queries.shape[0], device=batch_queries.device, dtype=torch.bool
            )
            if matched_src.numel() > 0:
                matched_mask[matched_src] = True

            target_boxes = target["boxes"]
            if target_boxes.numel() == 0:
                background_mask = ~matched_mask
            else:
                pred_boxes_xyxy = box_cxcywh_to_xyxy(batch_boxes.detach().float())
                target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes.detach().float())
                iou_matrix, _ = box_iou(pred_boxes_xyxy, target_boxes_xyxy)
                max_iou = iou_matrix.max(dim=1).values
                background_mask = (~matched_mask) & (max_iou < self.bg_iou_threshold)

            if matched_mask.any():
                fg_embeddings.append(batch_queries[matched_mask])
            if background_mask.any():
                bg_embeddings.append(batch_queries[background_mask])

        feature_dim = query_embeddings.shape[-1]
        if fg_embeddings:
            foreground_embeddings = torch.cat(fg_embeddings, dim=0)
        else:
            foreground_embeddings = query_embeddings.new_empty((0, feature_dim))

        if bg_embeddings:
            background_embeddings = torch.cat(bg_embeddings, dim=0)
        else:
            background_embeddings = query_embeddings.new_empty((0, feature_dim))

        return foreground_embeddings, background_embeddings
