from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """Lightweight EMA prototype memory for background embeddings."""

    def __init__(self, bank_size: int = 64, momentum: float = 0.9, eps: float = 1e-6):
        super().__init__()
        if bank_size <= 0:
            raise ValueError("bank_size must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")

        self.bank_size = int(bank_size)
        self.momentum = float(momentum)
        self.eps = float(eps)

        # Keep the bank as buffers so it follows the module device and can be checkpointed.
        self.register_buffer("prototypes", torch.empty(0))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Fresh models start with an empty placeholder buffer, but checkpoints may already
        # contain a fully materialized bank. Resize the local buffer first so the default
        # loader can copy the saved tensor without a shape mismatch.
        prototypes_key = prefix + "prototypes"
        if prototypes_key in state_dict:
            saved_prototypes = state_dict[prototypes_key]
            if self.prototypes.shape != saved_prototypes.shape:
                self._buffers["prototypes"] = self.prototypes.new_zeros(saved_prototypes.shape)
                if saved_prototypes.numel() > 0 and self.bank_size != saved_prototypes.shape[0]:
                    self.bank_size = int(saved_prototypes.shape[0])

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def num_active(self) -> int:
        if self.prototypes.numel() == 0:
            return 0
        return min(int(self.filled.item()), self.bank_size)

    def active_prototypes(self) -> torch.Tensor:
        if self.prototypes.numel() == 0:
            return self.prototypes.new_empty((0, 0))
        return self.prototypes[: self.num_active]

    def _ensure_initialized(self, feature_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        needs_init = self.prototypes.numel() == 0 or self.prototypes.shape[-1] != feature_dim
        if needs_init:
            self.prototypes = torch.zeros(self.bank_size, feature_dim, device=device, dtype=dtype)
            self.filled = torch.zeros((), device=device, dtype=torch.long)

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor) -> None:
        if embeddings is None or embeddings.numel() == 0:
            return

        embeddings = embeddings.detach()
        if embeddings.dim() != 2:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])

        embeddings = F.normalize(embeddings.float(), dim=-1, eps=self.eps)
        self._ensure_initialized(embeddings.shape[-1], embeddings.device, embeddings.dtype)

        active_count = self.num_active
        remaining = embeddings

        # Fill empty slots first.
        if active_count < self.bank_size:
            fill_count = min(self.bank_size - active_count, remaining.shape[0])
            if fill_count > 0:
                start = active_count
                end = start + fill_count
                self.prototypes[start:end] = remaining[:fill_count]
                self.filled.add_(fill_count)
                active_count = self.num_active
                remaining = remaining[fill_count:]

        if remaining.numel() == 0 or active_count == 0:
            return

        active = F.normalize(self.active_prototypes().float(), dim=-1, eps=self.eps)
        similarities = remaining @ active.t()
        assignments = similarities.argmax(dim=1)

        # EMA update each selected prototype with the mean of its assigned embeddings.
        for prototype_idx in torch.unique(assignments).tolist():
            mask = assignments == prototype_idx
            cluster_mean = remaining[mask].mean(dim=0)
            updated = self.momentum * self.prototypes[prototype_idx].float()
            updated = updated + (1.0 - self.momentum) * cluster_mean
            self.prototypes[prototype_idx] = F.normalize(updated, dim=-1, eps=self.eps)
