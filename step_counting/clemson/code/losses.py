import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHybridLoss(nn.Module):
    """Hybrid loss that blends dense sequence BCE with subject-level count MSE."""

    def __init__(
        self,
        label_width=3,
        alpha_max=0.15,
        warmup_ratio=0.2,
        ramp_epochs=60,
        eps=1.0,
        count_mode="mass",
        verbose=False,
    ):
        super().__init__()
        self.label_width = float(label_width)
        self.alpha_max = float(alpha_max)
        self.warmup_ratio = float(warmup_ratio)
        self.ramp_epochs = int(ramp_epochs)
        self.eps = float(eps)
        self.count_mode = count_mode
        self.verbose = verbose
        # Per-batch components of the most recent forward(), for logging.
        self.last_components = {}

    def _get_alpha(self, epoch, total_epochs):
        """Ramp the count-loss weight after the warmup stage."""
        warmup_epochs = int(total_epochs * self.warmup_ratio)

        if epoch < warmup_epochs:
            return 0.0

        ramp_start = warmup_epochs
        ramp_end = warmup_epochs + self.ramp_epochs

        if epoch >= ramp_end:
            return self.alpha_max

        progress = (epoch - ramp_start) / max(1, self.ramp_epochs)
        return self.alpha_max * progress

    def _predict_count(self, pred):
        """Convert dense probabilities to differentiable window-level counts."""
        if self.count_mode == "switch":
            prev = F.pad(pred[:, :-1], (1, 0), value=0.0)
            return F.relu(pred - prev).sum(dim=1)
        return pred.sum(dim=1) / self.label_width

    def _subject_relative_mse(self, pred, chunk_counts, subj_ids):
        """
        Compute relative MSE after aggregating window predictions per subject.

        pred:         (B, T)
        chunk_counts: (B, n_chunks)
        subj_ids:     (B,)
        """
        pred_count = self._predict_count(pred)
        true_count = chunk_counts.sum(dim=1)

        uniq = torch.unique(subj_ids)
        losses = []

        for s in uniq:
            mask = (subj_ids == s)
            pred_sum = pred_count[mask].sum()
            true_sum = true_count[mask].sum()
            loss_s = (pred_sum - true_sum) ** 2 / (true_sum + self.eps)
            losses.append(loss_s)

        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        return torch.stack(losses).mean()

    def forward(self, pred, y, chunk_counts, subj_ids, epoch, total_epochs):
        """
        pred:         (B, T)
        y:            (B, T)
        chunk_counts: (B, n_chunks)
        subj_ids:     (B,)
        """
        pred = pred.float()
        y = y.float()
        chunk_counts = chunk_counts.float()
        subj_ids = subj_ids.long()

        # Dense step-localization supervision.
        seq_loss = F.binary_cross_entropy(pred, y, reduction="mean")
        # Subject-level supervision stabilizes the total step-count objective.
        subject_rel_mse = self._subject_relative_mse(pred, chunk_counts, subj_ids)

        alpha = self._get_alpha(epoch, total_epochs)
        loss = (1.0 - alpha) * seq_loss + alpha * subject_rel_mse
        self.last_components = {
            "alpha": float(alpha),
            "bce": float(seq_loss.detach()),
            "count_mse": float(subject_rel_mse.detach()),
        }

        if self.verbose and (epoch % 10 == 0):
            print(
                f"[Loss] epoch={epoch} | "
                f"alpha={alpha:.4f} | "
                f"count_mode={self.count_mode} | "
                f"seq={seq_loss.item():.6f} | "
                f"subject_rel_mse={subject_rel_mse.item():.6f}"
            )

        return loss


class DenseBCELoss(nn.Module):
    """Pure dense sequence BCE loss without count-level supervision."""

    def forward(self, pred, y, chunk_counts, subj_ids, epoch, total_epochs):
        return F.binary_cross_entropy(pred.float(), y.float(), reduction="mean")


class DenseMSELoss(nn.Module):
    """Pure dense sequence MSE loss without count-level supervision."""

    def forward(self, pred, y, chunk_counts, subj_ids, epoch, total_epochs):
        return F.mse_loss(pred.float(), y.float(), reduction="mean")
