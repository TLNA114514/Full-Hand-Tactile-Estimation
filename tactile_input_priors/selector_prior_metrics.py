"""Streaming contact metrics for prior-aware frozen selector experiments."""

from __future__ import annotations

from typing import Dict, Sequence

import torch


def _average_precision(positive: torch.Tensor, negative: torch.Tensor) -> float:
    positive = positive.double().flip(0)
    negative = negative.double().flip(0)
    total_positive = positive.sum()
    if float(total_positive) <= 0.0:
        return float("nan")
    true_positive = positive.cumsum(0)
    false_positive = negative.cumsum(0)
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    return float((precision * positive / total_positive).sum())


def _high_precision_recall(
    positive: torch.Tensor, negative: torch.Tensor, minimum_precision: float
) -> float:
    positive = positive.double().flip(0)
    negative = negative.double().flip(0)
    total_positive = positive.sum()
    if float(total_positive) <= 0.0:
        return float("nan")
    true_positive = positive.cumsum(0)
    false_positive = negative.cumsum(0)
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / total_positive
    eligible = precision >= float(minimum_precision)
    return float(recall[eligible].max()) if bool(eligible.any()) else 0.0


class ContactMetricAccumulator:
    def __init__(
        self,
        *,
        bins: int = 2048,
        no_contact_max: float = 0.02,
        contact_min: float = 0.10,
        false_high_precision_targets: Sequence[float] = (
            0.50,
            0.60,
            0.70,
            0.80,
            0.85,
            0.90,
        ),
        false_high_topk: Sequence[int] = (1, 4, 16, 64),
    ):
        self.bins = int(bins)
        self.no_contact_max = float(no_contact_max)
        self.contact_min = float(contact_min)
        self.false_high_precision_targets = tuple(
            float(value) for value in false_high_precision_targets
        )
        self.false_high_topk_values = tuple(int(value) for value in false_high_topk)
        if any(not 0.0 < value <= 1.0 for value in self.false_high_precision_targets):
            raise ValueError("False-high precision targets must be in (0,1]")
        if any(value < 1 for value in self.false_high_topk_values):
            raise ValueError("False-high top-k values must be positive")
        self.reset()

    def reset(self) -> None:
        self.contact_positive = torch.zeros(self.bins, dtype=torch.float64)
        self.contact_negative = torch.zeros(self.bins, dtype=torch.float64)
        self.false_high_positive = torch.zeros(self.bins, dtype=torch.float64)
        self.false_high_negative = torch.zeros(self.bins, dtype=torch.float64)
        self.scalars = torch.zeros(9, dtype=torch.float64)
        # Columns are selected count and true-positive count. Recall uses the
        # same global positive denominator as the histogram metrics.
        self.false_high_topk = torch.zeros(
            (len(self.false_high_topk_values), 2), dtype=torch.float64
        )

    def _histogram(
        self,
        score: torch.Tensor,
        label: torch.Tensor,
        positive_hist: torch.Tensor,
        negative_hist: torch.Tensor,
    ) -> None:
        device = score.device
        indices = torch.clamp(
            (score.detach().float() * (self.bins - 1)).long(),
            0,
            self.bins - 1,
        )
        labels = label.detach().bool().to(device=device)
        positive_hist.add_(
            torch.bincount(indices[labels], minlength=self.bins)
            .to(dtype=torch.float64)
            .cpu()
        )
        negative_hist.add_(
            torch.bincount(indices[~labels], minlength=self.bins)
            .to(dtype=torch.float64)
            .cpu()
        )

    def update(
        self,
        contact_logits: torch.Tensor,
        target: torch.Tensor,
        palm_mask: torch.Tensor,
        has_tactile: torch.Tensor,
        *,
        base_pressure: torch.Tensor,
        false_high_logits: torch.Tensor | None = None,
    ) -> None:
        logits = contact_logits.detach().float()
        target = target.detach().float()
        palm = palm_mask.detach().float() > 0.5
        has = has_tactile.detach().reshape(-1) > 0.5
        valid = palm & has[:, None]
        clear = (target <= self.no_contact_max) | (target >= self.contact_min)
        eligible = valid & clear
        if not bool(eligible.any()):
            return
        labels = target >= self.contact_min
        probability = torch.sigmoid(logits)
        selected_probability = probability[eligible]
        selected_label = labels[eligible]
        self._histogram(
            selected_probability,
            selected_label,
            self.contact_positive,
            self.contact_negative,
        )
        prediction = selected_probability >= 0.5
        count = eligible.sum().double()
        nll = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[eligible], selected_label.float(), reduction="sum"
        ).double()
        brier = (selected_probability - selected_label.float()).square().sum().double()
        intersection = (prediction & selected_label).sum().double()
        union = (prediction | selected_label).sum().double()
        candidate_count = logits.new_zeros((), dtype=torch.float64)
        false_high_count = logits.new_zeros((), dtype=torch.float64)

        candidate = valid & (base_pressure.detach().float() >= self.contact_min)
        if bool(candidate.any()):
            false_high_label = target <= self.no_contact_max
            if false_high_logits is None:
                full_noncontact_score = torch.sigmoid(-logits)
            else:
                if false_high_logits.shape != contact_logits.shape:
                    raise ValueError(
                        "false_high_logits must match contact_logits shape"
                    )
                full_noncontact_score = torch.sigmoid(
                    false_high_logits.detach().float()
                )
            noncontact_score = full_noncontact_score[candidate]
            self._histogram(
                noncontact_score,
                false_high_label[candidate],
                self.false_high_positive,
                self.false_high_negative,
            )
            if self.false_high_topk_values:
                maximum_k = min(max(self.false_high_topk_values), candidate.shape[1])
                candidate_count = candidate.sum(dim=1)
                ranked_indices = torch.topk(
                    full_noncontact_score.masked_fill(~candidate, -1.0),
                    k=maximum_k,
                    dim=1,
                    sorted=True,
                ).indices
                ranked_positive = false_high_label.gather(1, ranked_indices)
                ranks = torch.arange(maximum_k, device=candidate.device)[None]
                topk_updates = []
                for topk in self.false_high_topk_values:
                    selected = ranks < candidate_count.clamp_max(topk)[:, None]
                    topk_updates.append(
                        torch.stack(
                            (selected.sum(), (selected & ranked_positive).sum())
                        )
                    )
                self.false_high_topk += torch.stack(topk_updates).double().cpu()
            candidate_count = candidate.sum().double()
            false_high_count = false_high_label[candidate].sum().double()
        valid_residual = logits[valid]
        self.scalars += torch.stack(
            (
                count,
                nll,
                brier,
                intersection,
                union,
                candidate_count,
                false_high_count,
                valid_residual.sum().double(),
                valid.sum().double(),
            )
        ).cpu()

    def synchronize(self, device: torch.device) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        values = torch.cat(
            (
                self.contact_positive,
                self.contact_negative,
                self.false_high_positive,
                self.false_high_negative,
                self.scalars,
                self.false_high_topk.flatten(),
            )
        ).to(device)
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values = values.cpu()
        offset = 0
        for name in (
            "contact_positive",
            "contact_negative",
            "false_high_positive",
            "false_high_negative",
        ):
            setattr(self, name, values[offset : offset + self.bins].clone())
            offset += self.bins
        self.scalars = values[offset : offset + len(self.scalars)].clone()
        offset += len(self.scalars)
        self.false_high_topk = values[offset:].reshape_as(self.false_high_topk).clone()

    def false_high_pr_curve(self) -> list[Dict[str, float]]:
        """Return the histogram PR curve from strictest to loosest threshold."""

        positive = self.false_high_positive.double().flip(0)
        negative = self.false_high_negative.double().flip(0)
        true_positive = positive.cumsum(0)
        false_positive = negative.cumsum(0)
        selected = true_positive + false_positive
        total_positive = positive.sum()
        total_candidates = selected[-1] if selected.numel() else selected.new_zeros(())
        rows: list[Dict[str, float]] = []
        for reverse_index in range(self.bins):
            if float(positive[reverse_index] + negative[reverse_index]) <= 0.0:
                continue
            original_index = self.bins - 1 - reverse_index
            rows.append(
                {
                    "score_threshold": float(original_index / max(self.bins - 1, 1)),
                    "precision": float(
                        true_positive[reverse_index]
                        / selected[reverse_index].clamp_min(1.0)
                    ),
                    "recall": float(
                        true_positive[reverse_index]
                        / total_positive.clamp_min(1.0)
                    ),
                    "selected_count": float(selected[reverse_index]),
                    "selected_fraction": float(
                        selected[reverse_index] / total_candidates.clamp_min(1.0)
                    ),
                }
            )
        return rows

    def summary(self) -> Dict[str, float]:
        count, nll, brier, intersection, union = self.scalars[:5]
        total_hist = self.contact_positive + self.contact_negative
        calibration_error = 0.0
        total = float(total_hist.sum())
        if total > 0:
            centers = (
                torch.arange(self.bins, dtype=torch.float64) + 0.5
            ) / self.bins
            empirical = self.contact_positive / total_hist.clamp_min(1.0)
            calibration_error = float(
                (total_hist / total * (empirical - centers).abs()).sum()
            )
        summary = {
            "contact_ap": _average_precision(
                self.contact_positive, self.contact_negative
            ),
            "contact_iou_at_0.5": float(intersection / union.clamp_min(1.0)),
            "contact_nll": float(nll / count.clamp_min(1.0)),
            "contact_brier": float(brier / count.clamp_min(1.0)),
            "contact_ece": calibration_error,
            "false_high_candidate_ap": _average_precision(
                self.false_high_positive, self.false_high_negative
            ),
            "false_high_candidate_count": float(self.scalars[5]),
            "false_high_positive_count": float(self.scalars[6]),
            "false_high_candidate_prevalence": float(
                self.scalars[6] / self.scalars[5].clamp_min(1.0)
            ),
            "contact_logit_mean": float(
                self.scalars[7] / self.scalars[8].clamp_min(1.0)
            ),
        }
        for minimum_precision in self.false_high_precision_targets:
            key = f"false_high_recall_at_precision_{minimum_precision:g}"
            summary[key] = _high_precision_recall(
                self.false_high_positive,
                self.false_high_negative,
                minimum_precision,
            )
        for index, topk in enumerate(self.false_high_topk_values):
            selected, true_positive = self.false_high_topk[index]
            summary[f"false_high_top{topk}_precision"] = float(
                true_positive / selected.clamp_min(1.0)
            )
            summary[f"false_high_top{topk}_recall"] = float(
                true_positive / self.scalars[6].clamp_min(1.0)
            )
            summary[f"false_high_top{topk}_selected_count"] = float(selected)
        return summary


class SequenceContactAPAccumulator:
    """Low-memory per-sequence AP using fixed probability bins."""

    def __init__(
        self,
        *,
        bins: int = 256,
        no_contact_max: float = 0.02,
        contact_min: float = 0.10,
    ):
        self.bins = int(bins)
        self.no_contact_max = float(no_contact_max)
        self.contact_min = float(contact_min)
        self.histograms: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self) -> None:
        self.histograms.clear()

    def update(
        self,
        contact_logits: torch.Tensor,
        target: torch.Tensor,
        palm_mask: torch.Tensor,
        has_tactile: torch.Tensor,
        sequence_keys: Sequence[str],
    ) -> None:
        probabilities = torch.sigmoid(contact_logits.detach().float())
        target = target.detach().float().to(device=probabilities.device)
        palm = palm_mask.detach().to(device=probabilities.device) > 0.5
        has = has_tactile.detach().reshape(-1).to(device=probabilities.device) > 0.5
        if len(sequence_keys) != probabilities.shape[0]:
            raise ValueError("sequence_key count does not match batch size")
        keys: list[str] = []
        key_to_index: dict[str, int] = {}
        sample_group = []
        for raw_key in sequence_keys:
            key = str(raw_key)
            if key not in key_to_index:
                key_to_index[key] = len(keys)
                keys.append(key)
            sample_group.append(key_to_index[key])

        valid = palm & has[:, None] & (
            (target <= self.no_contact_max) | (target >= self.contact_min)
        )
        labels = target >= self.contact_min
        bin_index = torch.clamp(
            (probabilities * (self.bins - 1)).long(), 0, self.bins - 1
        )
        group = torch.as_tensor(
            sample_group, device=probabilities.device, dtype=torch.long
        )[:, None]
        grouped_bins = group * self.bins + bin_index
        flat_size = len(keys) * self.bins
        positive = torch.bincount(
            grouped_bins[valid & labels], minlength=flat_size
        ).reshape(len(keys), self.bins)
        negative = torch.bincount(
            grouped_bins[valid & ~labels], minlength=flat_size
        ).reshape(len(keys), self.bins)
        positive = positive.to(dtype=torch.float64).cpu()
        negative = negative.to(dtype=torch.float64).cpu()
        for index, key in enumerate(keys):
            if not bool((positive[index] + negative[index]).any()):
                continue
            if key not in self.histograms:
                self.histograms[key] = (
                    torch.zeros(self.bins, dtype=torch.float64),
                    torch.zeros(self.bins, dtype=torch.float64),
                )
            self.histograms[key][0].add_(positive[index])
            self.histograms[key][1].add_(negative[index])

    def synchronize(self) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world_size = torch.distributed.get_world_size()
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, self.histograms)
        merged: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for shard in gathered:
            for key, (positive, negative) in shard.items():
                if key not in merged:
                    merged[key] = (
                        torch.zeros(self.bins, dtype=torch.float64),
                        torch.zeros(self.bins, dtype=torch.float64),
                    )
                merged[key][0].add_(positive)
                merged[key][1].add_(negative)
        self.histograms = merged

    def summary(self) -> Dict[str, float]:
        values = [
            _average_precision(positive, negative)
            for positive, negative in self.histograms.values()
        ]
        values = [value for value in values if value == value]
        return {
            "sequence_macro_contact_ap": (
                float(sum(values) / len(values)) if values else float("nan")
            ),
            "sequence_count": float(len(values)),
        }
