"""Streaming metrics for diagnostic temporal down/hold/up selectors."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.distributed as dist


ACTION_NAMES = ("down", "hold", "up")


def _binary_histogram_metrics(positive: torch.Tensor, negative: torch.Tensor) -> dict[str, float]:
    positive = positive.double().flip(0)
    negative = negative.double().flip(0)
    total_positive = float(positive.sum())
    total_negative = float(negative.sum())
    true_positive = positive.cumsum(0)
    false_positive = negative.cumsum(0)
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / max(total_positive, 1.0)
    recall_increment = torch.diff(
        torch.cat((recall.new_zeros(1), recall)), dim=0
    ).clamp_min(0.0)
    average_precision = float((precision * recall_increment).sum())
    if total_positive > 0.0 and total_negative > 0.0:
        true_positive_rate = true_positive / total_positive
        false_positive_rate = false_positive / total_negative
        auroc = float(
            torch.trapz(
                torch.cat((true_positive_rate.new_zeros(1), true_positive_rate)),
                torch.cat((false_positive_rate.new_zeros(1), false_positive_rate)),
            )
        )
    else:
        auroc = math.nan
    return {
        "average_precision": average_precision,
        "auroc": auroc,
        "positive_fraction": total_positive / max(total_positive + total_negative, 1.0),
    }


class BinaryScoreMetricAccumulator:
    """Streaming binary ranking metrics without retaining vertex predictions."""

    def __init__(self, *, bins: int = 4096):
        self.bins = int(bins)
        if self.bins < 16:
            raise ValueError("Binary score metric bins must be at least 16")
        self.histogram = torch.zeros(2, self.bins, dtype=torch.float64)
        self.score_sum = torch.zeros(2, dtype=torch.float64)
        self.nonfinite_count = torch.zeros((), dtype=torch.float64)

    def _ensure_device(self, device: torch.device) -> None:
        if self.histogram.device != device:
            self.histogram = self.histogram.to(device=device, non_blocking=True)
            self.score_sum = self.score_sum.to(device=device, non_blocking=True)
            self.nonfinite_count = self.nonfinite_count.to(
                device=device, non_blocking=True
            )

    @torch.no_grad()
    def update(
        self,
        score: torch.Tensor,
        positive: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        if score.shape != positive.shape:
            raise ValueError("Binary score and label tensors must have matching shapes")
        if valid is None:
            valid = torch.ones_like(positive, dtype=torch.bool)
        elif valid.shape != score.shape:
            raise ValueError("Binary score valid mask must match the score shape")
        mask = valid.bool().reshape(-1)
        selected_score = score.detach().float().reshape(-1)[mask]
        selected_positive = positive.detach().bool().reshape(-1)[mask]
        self._ensure_device(selected_score.device)
        finite = torch.isfinite(selected_score)
        self.nonfinite_count += (~finite).sum().double()
        selected_score = torch.nan_to_num(
            selected_score, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        bin_index = (selected_score * self.bins).long().clamp(0, self.bins - 1)
        for label in (False, True):
            current = selected_positive == label
            counts = torch.bincount(bin_index[current], minlength=self.bins)
            self.histogram[int(label)] += counts.double()
            self.score_sum[int(label)] += selected_score[current].double().sum()

    def synchronize(self, device: torch.device) -> None:
        for name in ("histogram", "score_sum", "nonfinite_count"):
            value = getattr(self, name)
            if value.device != device:
                value = value.to(device=device, non_blocking=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(value)
            setattr(self, name, value.cpu())

    def summary(self) -> dict[str, float]:
        if float(self.nonfinite_count) > 0.0:
            raise FloatingPointError(
                "Binary ranking scores contained "
                f"{int(self.nonfinite_count.item())} NaN or Inf values"
            )
        positive = self.histogram[1]
        negative = self.histogram[0]
        result = _binary_histogram_metrics(positive, negative)
        positive_count = float(positive.sum())
        negative_count = float(negative.sum())
        result.update(
            {
                "sample_count": positive_count + negative_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_score_mean": float(self.score_sum[1])
                / max(positive_count, 1.0),
                "negative_score_mean": float(self.score_sum[0])
                / max(negative_count, 1.0),
            }
        )
        return result

    def pr_curve_rows(self) -> list[dict[str, float]]:
        positive = self.histogram[1].double().flip(0)
        negative = self.histogram[0].double().flip(0)
        true_positive = positive.cumsum(0)
        false_positive = negative.cumsum(0)
        total_positive = float(positive.sum())
        rows = []
        for reverse_bin in range(self.bins):
            original_bin = self.bins - 1 - reverse_bin
            selected = true_positive[reverse_bin] + false_positive[reverse_bin]
            rows.append(
                {
                    "score_threshold": original_bin / self.bins,
                    "selected_count": float(selected),
                    "precision": float(
                        true_positive[reverse_bin] / selected.clamp_min(1.0)
                    ),
                    "recall": float(
                        true_positive[reverse_bin] / max(total_positive, 1.0)
                    ),
                }
            )
        return rows

    def budget_rows(
        self,
        coverages: Iterable[float] = (1e-5, 1e-4, 1e-3, 1e-2),
    ) -> list[dict[str, float]]:
        """Return thresholded points with approximately matched global coverage."""

        positive = self.histogram[1].double().flip(0)
        negative = self.histogram[0].double().flip(0)
        true_positive = positive.cumsum(0)
        false_positive = negative.cumsum(0)
        selected = true_positive + false_positive
        total = max(float(selected[-1]) if selected.numel() else 0.0, 1.0)
        total_positive = max(float(positive.sum()), 1.0)
        rows = []
        for requested in coverages:
            requested = float(requested)
            if not 0.0 < requested <= 1.0:
                raise ValueError("Binary score budget coverages must lie in (0,1]")
            target = requested * total
            matches = torch.nonzero(selected >= target, as_tuple=False).flatten()
            reverse_bin = int(matches[0]) if matches.numel() else self.bins - 1
            count = float(selected[reverse_bin])
            original_bin = self.bins - 1 - reverse_bin
            rows.append(
                {
                    "requested_coverage": requested,
                    "score_threshold": original_bin / self.bins,
                    "selected_count": count,
                    "actual_coverage": count / total,
                    "precision": float(
                        true_positive[reverse_bin]
                        / selected[reverse_bin].clamp_min(1.0)
                    ),
                    "recall": float(true_positive[reverse_bin] / total_positive),
                }
            )
        return rows


class SequenceBinaryScoreMetricAccumulator:
    """Per-sequence score histograms for clustered paired comparisons.

    The compact histogram is deliberately lower resolution than the primary
    micro metric. It is used for sequence-macro AP and clustered uncertainty,
    not as a replacement for the full-resolution headline metric.
    """

    def __init__(self, sequence_count: int, *, bins: int = 256):
        self.sequence_count = int(sequence_count)
        self.bins = int(bins)
        if self.sequence_count <= 0:
            raise ValueError("Sequence histogram requires at least one sequence")
        if self.bins < 32:
            raise ValueError("Sequence histogram bins must be at least 32")
        self.histogram = torch.zeros(
            self.sequence_count,
            2,
            self.bins,
            dtype=torch.float64,
        )
        self.nonfinite_count = torch.zeros((), dtype=torch.float64)

    def _ensure_device(self, device: torch.device) -> None:
        if self.histogram.device != device:
            self.histogram = self.histogram.to(device=device, non_blocking=True)
            self.nonfinite_count = self.nonfinite_count.to(
                device=device, non_blocking=True
            )

    @torch.no_grad()
    def update(
        self,
        score: torch.Tensor,
        positive: torch.Tensor,
        sequence_id: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        if score.shape != positive.shape or score.ndim != 2:
            raise ValueError("Sequence score and label tensors must be matching [B,N]")
        if sequence_id.shape != (score.shape[0],):
            raise ValueError("sequence_id must contain one value per score row")
        if valid is None:
            valid = torch.ones_like(positive, dtype=torch.bool)
        elif valid.shape != score.shape:
            raise ValueError("Sequence score valid mask must match the score shape")
        sequence = sequence_id.detach().long().to(score.device)
        sequence_valid = (sequence >= 0) & (sequence < self.sequence_count)
        mask = valid.bool() & sequence_valid[:, None]
        if not bool(mask.any()):
            return
        expanded_sequence = sequence[:, None].expand_as(score)[mask]
        selected_score = score.detach().float()[mask]
        selected_positive = positive.detach().bool()[mask]
        self._ensure_device(selected_score.device)
        finite = torch.isfinite(selected_score)
        self.nonfinite_count += (~finite).sum().double()
        selected_score = torch.nan_to_num(
            selected_score, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        score_bin = (selected_score * self.bins).long().clamp(0, self.bins - 1)
        flat_index = (
            (expanded_sequence * 2 + selected_positive.long()) * self.bins
            + score_bin
        )
        self.histogram.view(-1).scatter_add_(
            0,
            flat_index,
            torch.ones_like(flat_index, dtype=self.histogram.dtype),
        )

    def synchronize(self, device: torch.device) -> None:
        for name in ("histogram", "nonfinite_count"):
            value = getattr(self, name)
            if value.device != device:
                value = value.to(device=device, non_blocking=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(value)
            setattr(self, name, value.cpu())
        if float(self.nonfinite_count) > 0.0:
            raise FloatingPointError(
                "Sequence ranking scores contained "
                f"{int(self.nonfinite_count.item())} NaN or Inf values"
            )

    def sequence_average_precision(self) -> tuple[torch.Tensor, torch.Tensor]:
        positive = self.histogram[:, 1].double().flip(1)
        negative = self.histogram[:, 0].double().flip(1)
        total_positive = positive.sum(dim=1)
        total_negative = negative.sum(dim=1)
        true_positive = positive.cumsum(dim=1)
        false_positive = negative.cumsum(dim=1)
        precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
        average_precision = (
            precision * positive / total_positive[:, None].clamp_min(1.0)
        ).sum(dim=1)
        eligible = (total_positive > 0.0) & (total_negative > 0.0)
        return average_precision, eligible

    def micro_average_precision(self) -> float:
        total = self.histogram.sum(dim=0)
        return _binary_histogram_metrics(total[1], total[0])["average_precision"]

    def budget_components(
        self, coverage: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, torch.Tensor]:
        coverage = float(coverage)
        if not 0.0 < coverage <= 1.0:
            raise ValueError("Sequence score coverage must lie in (0,1]")
        positive = self.histogram[:, 1].double().flip(1)
        negative = self.histogram[:, 0].double().flip(1)
        total_positive = positive.sum(dim=1)
        total_count = (positive + negative).sum(dim=1)
        selected = (positive + negative).sum(dim=0).cumsum(dim=0)
        total = max(float(selected[-1]) if selected.numel() else 0.0, 1.0)
        matches = torch.nonzero(selected >= coverage * total, as_tuple=False).flatten()
        reverse_bin = int(matches[0]) if matches.numel() else self.bins - 1
        true_positive = positive[:, : reverse_bin + 1].sum(dim=1)
        false_positive = negative[:, : reverse_bin + 1].sum(dim=1)
        actual_coverage = float(selected[reverse_bin]) / total
        threshold = (self.bins - 1 - reverse_bin) / self.bins
        return (
            true_positive,
            false_positive,
            total_positive,
            threshold,
            actual_coverage,
            total_count,
        )

class TemporalActionMetricAccumulator:
    """Bounded-memory class, calibration, and coverage statistics."""

    def __init__(
        self,
        *,
        bins: int = 1024,
        coverage_thresholds: Iterable[float] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    ):
        self.bins = int(bins)
        if self.bins < 16:
            raise ValueError("Temporal selector metric bins must be at least 16")
        self.coverage_thresholds = tuple(float(value) for value in coverage_thresholds)
        if any(value < 0.0 or value > 1.0 for value in self.coverage_thresholds):
            raise ValueError("Coverage thresholds must lie in [0,1]")
        self.reset()

    def reset(self) -> None:
        self.one_vs_rest = torch.zeros(3, 2, self.bins, dtype=torch.float64)
        self.confusion = torch.zeros(3, 3, dtype=torch.float64)
        self.calibration = torch.zeros(self.bins, 3, dtype=torch.float64)
        self.coverage = torch.zeros(len(self.coverage_thresholds), 3, dtype=torch.float64)
        self.loss = torch.zeros(2, dtype=torch.float64)

    @torch.no_grad()
    def update(
        self,
        probability: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        *,
        loss: torch.Tensor | None = None,
    ) -> None:
        if probability.shape != (*target.shape, 3):
            raise ValueError(
                f"probability/target shape mismatch: {tuple(probability.shape)} vs "
                f"{tuple(target.shape)}"
            )
        if valid is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        elif valid.ndim == 1:
            valid = valid[:, None].expand_as(target)
        if valid.shape != target.shape:
            raise ValueError("Temporal selector valid mask must match target anchors")
        mask = valid.bool().reshape(-1)
        if not bool(mask.any()):
            return
        probability = probability.detach().float().reshape(-1, 3)[mask].cpu()
        target = target.detach().long().reshape(-1)[mask].cpu()
        if not torch.isfinite(probability).all():
            raise FloatingPointError("Temporal selector probabilities are non-finite")
        prediction = probability.argmax(dim=-1)
        confidence = probability.max(dim=-1).values
        correct = prediction == target
        flat_confusion = target * 3 + prediction
        self.confusion += torch.bincount(flat_confusion, minlength=9).reshape(3, 3)
        for action in range(3):
            bins = (probability[:, action] * self.bins).long().clamp(0, self.bins - 1)
            positive = target == action
            self.one_vs_rest[action, 1] += torch.bincount(
                bins[positive], minlength=self.bins
            )
            self.one_vs_rest[action, 0] += torch.bincount(
                bins[~positive], minlength=self.bins
            )
        confidence_bin = (confidence * self.bins).long().clamp(0, self.bins - 1)
        self.calibration[:, 0] += torch.bincount(
            confidence_bin, minlength=self.bins
        )
        self.calibration[:, 1] += torch.bincount(
            confidence_bin, weights=confidence.double(), minlength=self.bins
        )
        self.calibration[:, 2] += torch.bincount(
            confidence_bin, weights=correct.double(), minlength=self.bins
        )
        for index, threshold in enumerate(self.coverage_thresholds):
            selected = confidence >= threshold
            self.coverage[index, 0] += selected.sum()
            self.coverage[index, 1] += correct[selected].sum()
            self.coverage[index, 2] += ((prediction[selected] != 1) & correct[selected]).sum()
        if loss is not None:
            self.loss[0] += float(loss.detach()) * int(mask.sum())
            self.loss[1] += int(mask.sum())

    def synchronize(self, device: torch.device) -> None:
        for name in ("one_vs_rest", "confusion", "calibration", "coverage", "loss"):
            value = getattr(self, name).to(device=device, non_blocking=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(value)
            setattr(self, name, value.cpu())

    def summary(self) -> dict[str, Any]:
        total = float(self.confusion.sum())
        diagonal = self.confusion.diag()
        per_class = {}
        aps = []
        aucs = []
        f1s = []
        for action, name in enumerate(ACTION_NAMES):
            binary = _binary_histogram_metrics(
                self.one_vs_rest[action, 1], self.one_vs_rest[action, 0]
            )
            true_positive = float(diagonal[action])
            predicted = float(self.confusion[:, action].sum())
            actual = float(self.confusion[action].sum())
            precision = true_positive / max(predicted, 1.0)
            recall = true_positive / max(actual, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            per_class[name] = {
                **binary,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "count": actual,
            }
            aps.append(binary["average_precision"])
            if math.isfinite(binary["auroc"]):
                aucs.append(binary["auroc"])
            f1s.append(f1)
        count = self.calibration[:, 0]
        confidence = self.calibration[:, 1] / count.clamp_min(1.0)
        accuracy = self.calibration[:, 2] / count.clamp_min(1.0)
        ece = float(((confidence - accuracy).abs() * count).sum() / count.sum().clamp_min(1.0))
        coverage_rows = []
        for threshold, values in zip(self.coverage_thresholds, self.coverage):
            selected = float(values[0])
            coverage_rows.append(
                {
                    "confidence_threshold": threshold,
                    "selected_count": selected,
                    "coverage": selected / max(total, 1.0),
                    "accuracy": float(values[1]) / max(selected, 1.0),
                    "correct_nonhold_fraction": float(values[2]) / max(selected, 1.0),
                }
            )
        return {
            "sample_count": total,
            "loss": float(self.loss[0] / self.loss[1].clamp_min(1.0)),
            "accuracy": float(diagonal.sum()) / max(total, 1.0),
            "macro_f1": sum(f1s) / len(f1s),
            "macro_average_precision": sum(aps) / len(aps),
            "macro_auroc": sum(aucs) / max(len(aucs), 1),
            "ece": ece,
            "per_class": per_class,
            "confusion": self.confusion.tolist(),
            "risk_coverage": coverage_rows,
        }

    def pr_curve_rows(self) -> list[dict[str, float | str]]:
        rows = []
        for action, name in enumerate(ACTION_NAMES):
            positive = self.one_vs_rest[action, 1].double().flip(0)
            negative = self.one_vs_rest[action, 0].double().flip(0)
            true_positive = positive.cumsum(0)
            false_positive = negative.cumsum(0)
            total_positive = float(positive.sum())
            for reverse_bin in range(self.bins):
                original_bin = self.bins - 1 - reverse_bin
                rows.append(
                    {
                        "action": name,
                        "probability_threshold": original_bin / self.bins,
                        "precision": float(
                            true_positive[reverse_bin]
                            / (true_positive[reverse_bin] + false_positive[reverse_bin]).clamp_min(1.0)
                        ),
                        "recall": float(true_positive[reverse_bin] / max(total_positive, 1.0)),
                    }
                )
        return rows

    def calibration_curve_rows(self) -> list[dict[str, float]]:
        rows = []
        for bin_index, values in enumerate(self.calibration):
            count = float(values[0])
            if count <= 0.0:
                continue
            rows.append(
                {
                    "confidence_low": bin_index / self.bins,
                    "confidence_high": (bin_index + 1) / self.bins,
                    "count": count,
                    "mean_confidence": float(values[1]) / count,
                    "accuracy": float(values[2]) / count,
                }
            )
        return rows
