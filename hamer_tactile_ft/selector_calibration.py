"""Streaming calibration utilities for canonical support selectors."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence


SELECTOR_CALIBRATION_VERSION = 1
SELECTOR_LOGIT_MIN = -24.0
SELECTOR_LOGIT_MAX = 24.0
SELECTOR_HISTOGRAM_BINS = 4096
SELECTOR_CORRECTION_MIN_PRECISION = 0.90


def selector_histogram_rows(output_count: int) -> int:
    """Two cumulative rows per output, two clear rows, and four base-error rows."""

    output_count = int(output_count)
    if output_count < 1:
        raise ValueError("output_count must be positive")
    return 2 * output_count + 6


def selector_histogram_layout(output_count: int) -> Dict[str, object]:
    output_count = int(output_count)
    cumulative = [
        {"positive": 2 * index, "negative": 2 * index + 1}
        for index in range(output_count)
    ]
    offset = 2 * output_count
    return {
        "cumulative": cumulative,
        "clear_positive": offset,
        "clear_negative": offset + 1,
        "false_high": offset + 2,
        "base_true_positive": offset + 3,
        "false_low": offset + 4,
        "base_true_negative": offset + 5,
    }


def logit_bin_lower_edge(
    index: int,
    *,
    bin_count: int = SELECTOR_HISTOGRAM_BINS,
    logit_min: float = SELECTOR_LOGIT_MIN,
    logit_max: float = SELECTOR_LOGIT_MAX,
) -> float:
    bin_width = (float(logit_max) - float(logit_min)) / int(bin_count)
    if int(index) >= int(bin_count):
        # A finite edge above the clamped histogram range represents "select none"
        # and remains JSON/provenance friendly.
        return float(logit_max) + bin_width
    if int(index) <= 0:
        return float(logit_min)
    return float(logit_min) + (
        (float(logit_max) - float(logit_min)) * int(index) / int(bin_count)
    )


def probability_from_logit(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-float(logit)))
    exponential = math.exp(float(logit))
    return exponential / (1.0 + exponential)


def _binary_metrics(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    return {
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "f1": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _descending_counts(
    positive_histogram: Sequence[float],
    negative_histogram: Sequence[float],
) -> List[Dict[str, float]]:
    if len(positive_histogram) != len(negative_histogram):
        raise ValueError("positive and negative histograms must have equal length")
    positive_total = float(sum(float(value) for value in positive_histogram))
    negative_total = float(sum(float(value) for value in negative_histogram))
    true_positive = 0.0
    false_positive = 0.0
    points = [
        {
            "index": len(positive_histogram),
            **_binary_metrics(0.0, 0.0, positive_total, negative_total),
        }
    ]
    for index in range(len(positive_histogram) - 1, -1, -1):
        true_positive += float(positive_histogram[index])
        false_positive += float(negative_histogram[index])
        points.append(
            {
                "index": index,
                **_binary_metrics(
                    true_positive,
                    false_positive,
                    positive_total - true_positive,
                    negative_total - false_positive,
                ),
            }
        )
    return points


def _average_precision(points: Sequence[Mapping[str, float]]) -> float:
    value = 0.0
    previous_recall = 0.0
    for point in points[1:]:
        recall = float(point["recall"])
        value += max(recall - previous_recall, 0.0) * float(point["precision"])
        previous_recall = recall
    return value


def _roc_auc(points: Sequence[Mapping[str, float]]) -> float:
    positive_total = float(points[-1]["tp"] + points[-1]["fn"])
    negative_total = float(points[-1]["fp"] + points[-1]["tn"])
    if positive_total <= 0.0 or negative_total <= 0.0:
        return float("nan")
    value = 0.0
    previous_fpr = 0.0
    previous_tpr = 0.0
    for point in points[1:]:
        false_positive_rate = float(point["fp"]) / negative_total
        true_positive_rate = float(point["tp"]) / positive_total
        value += (false_positive_rate - previous_fpr) * (
            true_positive_rate + previous_tpr
        ) * 0.5
        previous_fpr = false_positive_rate
        previous_tpr = true_positive_rate
    return value


def _best_iou_point(points: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    # Prefer precision and then the more conservative threshold on exact ties.
    return dict(
        max(
            points,
            key=lambda point: (
                float(point["iou"]),
                float(point["precision"]),
                int(point["index"]),
            ),
        )
    )


def _point_at_index(
    positive_histogram: Sequence[float],
    negative_histogram: Sequence[float],
    threshold_index: int,
) -> Dict[str, float]:
    bin_count = len(positive_histogram)
    threshold_index = min(max(int(threshold_index), 0), bin_count)
    true_positive = sum(float(value) for value in positive_histogram[threshold_index:])
    false_positive = sum(float(value) for value in negative_histogram[threshold_index:])
    positive_total = sum(float(value) for value in positive_histogram)
    negative_total = sum(float(value) for value in negative_histogram)
    return {
        "index": threshold_index,
        **_binary_metrics(
            true_positive,
            false_positive,
            positive_total - true_positive,
            negative_total - false_positive,
        ),
    }


def _threshold_fields(
    point: Mapping[str, float],
    *,
    bin_count: int,
    logit_min: float,
    logit_max: float,
) -> Dict[str, float]:
    logit = logit_bin_lower_edge(
        int(point["index"]),
        bin_count=bin_count,
        logit_min=logit_min,
        logit_max=logit_max,
    )
    return {
        "threshold_index": int(point["index"]),
        "threshold_logit": logit,
        "threshold_probability": probability_from_logit(logit),
    }


def _select_correction_thresholds(
    error_histograms: Mapping[str, Sequence[float]],
    *,
    minimum_precision: float,
    bin_count: int,
    logit_min: float,
    logit_max: float,
) -> Dict[str, Dict[str, float]]:
    false_high = [float(value) for value in error_histograms["false_high"]]
    base_true_positive = [
        float(value) for value in error_histograms["base_true_positive"]
    ]
    false_low = [float(value) for value in error_histograms["false_low"]]
    base_true_negative = [
        float(value) for value in error_histograms["base_true_negative"]
    ]

    false_high_total = max(sum(false_high), 1.0)
    false_low_total = max(sum(false_low), 1.0)
    down_false_high = 0.0
    down_true_positive = 0.0
    down_candidates = []
    for index in range(bin_count + 1):
        if index > 0:
            down_false_high += false_high[index - 1]
            down_true_positive += base_true_positive[index - 1]
        denominator = down_false_high + down_true_positive
        precision = down_false_high / max(denominator, 1.0)
        coverage = down_false_high / false_high_total
        if denominator > 0.0 and precision >= float(minimum_precision):
            down_candidates.append((coverage, precision, -index, index, denominator))
    if down_candidates:
        coverage, precision, _, down_index, denominator = max(down_candidates)
    else:
        down_index = 0
        coverage = precision = denominator = 0.0

    up_false_low = 0.0
    up_true_negative = 0.0
    up_candidates = []
    for index in range(bin_count, -1, -1):
        if index < bin_count:
            up_false_low += false_low[index]
            up_true_negative += base_true_negative[index]
        denominator_value = up_false_low + up_true_negative
        precision_value = up_false_low / max(denominator_value, 1.0)
        recovery = up_false_low / false_low_total
        if denominator_value > 0.0 and precision_value >= float(minimum_precision):
            up_candidates.append(
                (recovery, precision_value, index, index, denominator_value)
            )
    if up_candidates:
        recovery, up_precision, _, up_index, up_denominator = max(up_candidates)
    else:
        up_index = bin_count
        recovery = up_precision = up_denominator = 0.0

    down = {
        **_threshold_fields(
            {"index": down_index},
            bin_count=bin_count,
            logit_min=logit_min,
            logit_max=logit_max,
        ),
        "precision": float(precision),
        "false_high_coverage": float(coverage),
        "selected_count": float(denominator),
        "minimum_precision": float(minimum_precision),
        "average_precision": _average_precision(
            _descending_counts(
                list(reversed(false_high)),
                list(reversed(base_true_positive)),
            )
        ),
        "roc_auc": _roc_auc(
            _descending_counts(
                list(reversed(false_high)),
                list(reversed(base_true_positive)),
            )
        ),
    }
    up = {
        **_threshold_fields(
            {"index": up_index},
            bin_count=bin_count,
            logit_min=logit_min,
            logit_max=logit_max,
        ),
        "precision": float(up_precision),
        "false_low_recovery": float(recovery),
        "selected_count": float(up_denominator),
        "minimum_precision": float(minimum_precision),
        "average_precision": _average_precision(
            _descending_counts(false_low, base_true_negative)
        ),
        "roc_auc": _roc_auc(
            _descending_counts(false_low, base_true_negative)
        ),
    }
    return {"down": down, "up": up}


def summarize_selector_histograms(
    histogram: Sequence[Sequence[float]],
    thresholds: Iterable[float],
    *,
    contact_index: int,
    minimum_correction_precision: float = SELECTOR_CORRECTION_MIN_PRECISION,
    logit_min: float = SELECTOR_LOGIT_MIN,
    logit_max: float = SELECTOR_LOGIT_MAX,
) -> Dict[str, object]:
    thresholds = tuple(float(value) for value in thresholds)
    rows = [[float(value) for value in row] for row in histogram]
    if len(rows) != selector_histogram_rows(len(thresholds)):
        raise ValueError("selector histogram row count does not match thresholds")
    if not 0 <= int(contact_index) < len(thresholds):
        raise ValueError("contact_index lies outside selector thresholds")
    bin_count = len(rows[0])
    if bin_count < 2 or any(len(row) != bin_count for row in rows):
        raise ValueError("selector histogram must be a non-ragged 2D array")
    layout = selector_histogram_layout(len(thresholds))

    cumulative = []
    for threshold_index, threshold in enumerate(thresholds):
        indices = layout["cumulative"][threshold_index]
        positive = rows[indices["positive"]]
        negative = rows[indices["negative"]]
        points = _descending_counts(positive, negative)
        best = _best_iou_point(points)
        cumulative.append(
            {
                "target_threshold": float(threshold),
                **_threshold_fields(
                    best,
                    bin_count=bin_count,
                    logit_min=logit_min,
                    logit_max=logit_max,
                ),
                "metrics": {
                    name: float(best[name])
                    for name in ("iou", "precision", "recall", "f1", "accuracy")
                },
                "average_precision": _average_precision(points),
                "roc_auc": _roc_auc(points),
            }
        )

    clear_positive = rows[int(layout["clear_positive"])]
    clear_negative = rows[int(layout["clear_negative"])]
    clear_points = _descending_counts(clear_positive, clear_negative)
    clear_best = _best_iou_point(clear_points)
    calibrated_contact_index = int(clear_best["index"])
    contact_layout = layout["cumulative"][int(contact_index)]
    all_at_clear_threshold = _point_at_index(
        rows[contact_layout["positive"]],
        rows[contact_layout["negative"]],
        calibrated_contact_index,
    )
    cumulative[int(contact_index)] = {
        "target_threshold": float(thresholds[int(contact_index)]),
        **_threshold_fields(
            clear_best,
            bin_count=bin_count,
            logit_min=logit_min,
            logit_max=logit_max,
        ),
        "metrics": {
            name: float(all_at_clear_threshold[name])
            for name in ("iou", "precision", "recall", "f1", "accuracy")
        },
        "average_precision": cumulative[int(contact_index)]["average_precision"],
        "roc_auc": cumulative[int(contact_index)]["roc_auc"],
        "selected_on": "clear_iou",
    }

    error_histograms = {
        name: rows[int(layout[name])]
        for name in (
            "false_high",
            "base_true_positive",
            "false_low",
            "base_true_negative",
        )
    }
    return {
        "version": SELECTOR_CALIBRATION_VERSION,
        "bin_count": int(bin_count),
        "logit_min": float(logit_min),
        "logit_max": float(logit_max),
        "contact_index": int(contact_index),
        "thresholds": list(thresholds),
        "probability_thresholds": [
            float(item["threshold_probability"]) for item in cumulative
        ],
        "logit_thresholds": [float(item["threshold_logit"]) for item in cumulative],
        "cumulative": cumulative,
        "contact_clear": {
            **_threshold_fields(
                clear_best,
                bin_count=bin_count,
                logit_min=logit_min,
                logit_max=logit_max,
            ),
            "metrics": {
                name: float(clear_best[name])
                for name in ("iou", "precision", "recall", "f1", "accuracy")
            },
            "average_precision": _average_precision(clear_points),
            "roc_auc": _roc_auc(clear_points),
        },
        "correction": _select_correction_thresholds(
            error_histograms,
            minimum_precision=float(minimum_correction_precision),
            bin_count=bin_count,
            logit_min=logit_min,
            logit_max=logit_max,
        ),
    }


def histogram_threshold_index(
    probability: float,
    *,
    bin_count: int = SELECTOR_HISTOGRAM_BINS,
    logit_min: float = SELECTOR_LOGIT_MIN,
    logit_max: float = SELECTOR_LOGIT_MAX,
) -> int:
    probability = float(probability)
    if probability <= 0.0:
        return 0
    if probability >= 1.0:
        return int(bin_count)
    logit = math.log(probability / (1.0 - probability))
    raw_index = (
        (logit - float(logit_min))
        * int(bin_count)
        / (float(logit_max) - float(logit_min))
    )
    nearest = round(raw_index)
    index = (
        int(nearest)
        if abs(raw_index - nearest) <= 1e-7
        else math.ceil(raw_index)
    )
    return min(max(int(index), 0), int(bin_count))


def calibrated_counts(
    histogram: Sequence[Sequence[float]],
    thresholds: Iterable[float],
    probability_thresholds: Sequence[float],
    *,
    contact_index: int,
    threshold_indices: Sequence[int] | None = None,
    logit_min: float = SELECTOR_LOGIT_MIN,
    logit_max: float = SELECTOR_LOGIT_MAX,
) -> Dict[str, object]:
    thresholds = tuple(float(value) for value in thresholds)
    rows = [[float(value) for value in row] for row in histogram]
    if len(probability_thresholds) != len(thresholds):
        raise ValueError("calibrated selector thresholds do not match model outputs")
    bin_count = len(rows[0])
    layout = selector_histogram_layout(len(thresholds))
    if threshold_indices is None:
        indices = [
            histogram_threshold_index(
                probability,
                bin_count=bin_count,
                logit_min=logit_min,
                logit_max=logit_max,
            )
            for probability in probability_thresholds
        ]
    else:
        if len(threshold_indices) != len(thresholds):
            raise ValueError("calibrated selector indices do not match model outputs")
        indices = [
            min(max(int(index), 0), bin_count) for index in threshold_indices
        ]
    cumulative = []
    for output_index, target_threshold in enumerate(thresholds):
        pair = layout["cumulative"][output_index]
        point = _point_at_index(
            rows[pair["positive"]],
            rows[pair["negative"]],
            indices[output_index],
        )
        cumulative.append(
            {
                "target_threshold": target_threshold,
                "threshold_probability": float(probability_thresholds[output_index]),
                "metrics": {
                    name: float(point[name])
                    for name in ("iou", "precision", "recall", "f1", "accuracy")
                },
            }
        )
    clear_point = _point_at_index(
        rows[int(layout["clear_positive"])],
        rows[int(layout["clear_negative"])],
        indices[int(contact_index)],
    )
    return {
        "probability_thresholds": [float(value) for value in probability_thresholds],
        "cumulative": cumulative,
        "contact_clear": {
            "threshold_probability": float(probability_thresholds[int(contact_index)]),
            "metrics": {
                name: float(clear_point[name])
                for name in ("iou", "precision", "recall", "f1", "accuracy")
            },
        },
    }


def calibrated_correction_counts(
    histogram: Sequence[Sequence[float]],
    thresholds: Iterable[float],
    correction: Mapping[str, Mapping[str, float]],
    *,
    logit_min: float = SELECTOR_LOGIT_MIN,
    logit_max: float = SELECTOR_LOGIT_MAX,
) -> Dict[str, Dict[str, float]]:
    thresholds = tuple(float(value) for value in thresholds)
    rows = [[float(value) for value in row] for row in histogram]
    layout = selector_histogram_layout(len(thresholds))
    bin_count = len(rows[0])
    down_index = int(correction["down"].get("threshold_index", -1))
    if down_index < 0:
        down_index = histogram_threshold_index(
            float(correction["down"]["threshold_probability"]),
            bin_count=bin_count,
            logit_min=logit_min,
            logit_max=logit_max,
        )
    up_index = int(correction["up"].get("threshold_index", -1))
    if up_index < 0:
        up_index = histogram_threshold_index(
            float(correction["up"]["threshold_probability"]),
            bin_count=bin_count,
            logit_min=logit_min,
            logit_max=logit_max,
        )
    down_index = min(max(down_index, 0), bin_count)
    up_index = min(max(up_index, 0), bin_count)
    false_high = rows[int(layout["false_high"])]
    base_true_positive = rows[int(layout["base_true_positive"])]
    false_low = rows[int(layout["false_low"])]
    base_true_negative = rows[int(layout["base_true_negative"])]
    selected_false_high = sum(false_high[:down_index])
    selected_true_positive = sum(base_true_positive[:down_index])
    selected_false_low = sum(false_low[up_index:])
    selected_true_negative = sum(base_true_negative[up_index:])
    return {
        "down": {
            "threshold_probability": float(
                correction["down"]["threshold_probability"]
            ),
            "precision": selected_false_high
            / max(selected_false_high + selected_true_positive, 1.0),
            "false_high_coverage": selected_false_high
            / max(sum(false_high), 1.0),
            "selected_count": selected_false_high + selected_true_positive,
        },
        "up": {
            "threshold_probability": float(
                correction["up"]["threshold_probability"]
            ),
            "precision": selected_false_low
            / max(selected_false_low + selected_true_negative, 1.0),
            "false_low_recovery": selected_false_low / max(sum(false_low), 1.0),
            "selected_count": selected_false_low + selected_true_negative,
        },
    }


def selector_threshold_curve(
    histogram: Sequence[Sequence[float]],
    thresholds: Iterable[float],
    *,
    contact_index: int,
    logit_min: float = SELECTOR_LOGIT_MIN,
    logit_max: float = SELECTOR_LOGIT_MAX,
) -> List[Dict[str, float]]:
    """Return the full clear-contact and correction threshold curve.

    This curve is diagnostic.  Formal test metrics must apply thresholds chosen
    on validation instead of selecting the best row from a test curve.
    """

    thresholds = tuple(float(value) for value in thresholds)
    rows = [[float(value) for value in row] for row in histogram]
    if len(rows) != selector_histogram_rows(len(thresholds)):
        raise ValueError("selector histogram row count does not match thresholds")
    if not 0 <= int(contact_index) < len(thresholds):
        raise ValueError("contact_index lies outside selector thresholds")
    bin_count = len(rows[0])
    if bin_count < 2 or any(len(row) != bin_count for row in rows):
        raise ValueError("selector histogram must be a non-ragged 2D array")

    layout = selector_histogram_layout(len(thresholds))
    clear_positive = rows[int(layout["clear_positive"])]
    clear_negative = rows[int(layout["clear_negative"])]
    contact_pair = layout["cumulative"][int(contact_index)]
    all_positive = rows[contact_pair["positive"]]
    all_negative = rows[contact_pair["negative"]]
    false_high = rows[int(layout["false_high"])]
    base_true_positive = rows[int(layout["base_true_positive"])]
    false_low = rows[int(layout["false_low"])]
    base_true_negative = rows[int(layout["base_true_negative"])]

    false_high_total = max(sum(false_high), 1.0)
    false_low_total = max(sum(false_low), 1.0)
    clear_points = {
        int(point["index"]): point
        for point in _descending_counts(clear_positive, clear_negative)
    }
    all_points = {
        int(point["index"]): point
        for point in _descending_counts(all_positive, all_negative)
    }
    up_false_low = [0.0] * (bin_count + 1)
    up_true_negative = [0.0] * (bin_count + 1)
    for index in range(bin_count - 1, -1, -1):
        up_false_low[index] = up_false_low[index + 1] + false_low[index]
        up_true_negative[index] = (
            up_true_negative[index + 1] + base_true_negative[index]
        )
    curve = []
    down_false_high = 0.0
    down_true_positive = 0.0
    for index in range(bin_count + 1):
        if index > 0:
            down_false_high += false_high[index - 1]
            down_true_positive += base_true_positive[index - 1]
        clear = clear_points[index]
        all_contact = all_points[index]
        threshold_logit = logit_bin_lower_edge(
            index,
            bin_count=bin_count,
            logit_min=logit_min,
            logit_max=logit_max,
        )
        curve.append(
            {
                "threshold_index": int(index),
                "threshold_logit": float(threshold_logit),
                "threshold_probability": probability_from_logit(threshold_logit),
                "clear_iou": float(clear["iou"]),
                "clear_precision": float(clear["precision"]),
                "clear_recall": float(clear["recall"]),
                "clear_f1": float(clear["f1"]),
                "all_iou": float(all_contact["iou"]),
                "all_precision": float(all_contact["precision"]),
                "all_recall": float(all_contact["recall"]),
                "down_precision": down_false_high
                / max(down_false_high + down_true_positive, 1.0),
                "false_high_coverage": down_false_high / false_high_total,
                "up_precision": up_false_low[index]
                / max(up_false_low[index] + up_true_negative[index], 1.0),
                "false_low_recovery": up_false_low[index] / false_low_total,
            }
        )
    return curve
