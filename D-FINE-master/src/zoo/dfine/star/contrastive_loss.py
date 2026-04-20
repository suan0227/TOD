from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_repulsion_loss(
    foreground_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    margin: float = 0.4,
) -> torch.Tensor:
    """Repulsive hinge loss against background prototypes."""
    fg = F.normalize(foreground_embeddings.float(), dim=-1)
    proto = F.normalize(prototypes.float(), dim=-1)
    similarity = fg @ proto.t()
    return F.relu(similarity - margin).mean()

