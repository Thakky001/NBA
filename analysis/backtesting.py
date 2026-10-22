"""Leakage-safe chronological splits and evaluation helpers for NBA forecasts."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple


def walk_forward_splits(dates: Sequence, min_train: int, validation_size: int,
                        test_size: int, step: int | None = None) -> List[Dict[str, List[int]]]:
    """Create expanding train/validation/test folds from already ordered rows."""
    n = len(dates)
    step = test_size if step is None else step
    if min_train < 1 or validation_size < 1 or test_size < 1 or step < 1:
        raise ValueError("split sizes and step must be positive")
    if any(dates[i] > dates[i + 1] for i in range(n - 1)):
        raise ValueError("dates must be sorted ascending before splitting")
    folds = []
    test_start = min_train + validation_size
    while test_start + test_size <= n:
        folds.append({
            "train": list(range(test_start - validation_size)),
            "validation": list(range(test_start - validation_size, test_start)),
            "test": list(range(test_start, test_start + test_size)),
        })
        test_start += step
    return folds


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> Dict[str, float]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted must be non-empty and the same length")
    errors = [float(p) - float(y) for y, p in zip(actual, predicted)]
    return {
        "mae": sum(abs(e) for e in errors) / len(errors),
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
    }


def probability_metrics(probabilities: Sequence[float], outcomes: Sequence[int]) -> Dict[str, float]:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be non-empty and the same length")
    if any(not 0 <= p <= 1 for p in probabilities) or any(y not in (0, 1) for y in outcomes):
        raise ValueError("probabilities must be in [0, 1] and outcomes must be binary")
    eps = 1e-15
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes)
    logloss = -sum(y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p))
                   for p, y in zip(probabilities, outcomes)) / len(outcomes)
    return {"brier": brier, "log_loss": logloss}


def calibration_bins(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> List[Dict[str, float]]:
    if bins < 1 or len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("provide matching non-empty forecasts and a positive bin count")
    groups = [[] for _ in range(bins)]
    for p, y in zip(probabilities, outcomes):
        if not 0 <= p <= 1 or y not in (0, 1):
            raise ValueError("probabilities must be in [0, 1] and outcomes must be binary")
        groups[min(int(p * bins), bins - 1)].append((p, y))
    return [{"count": len(g), "mean_probability": sum(p for p, _ in g) / len(g),
             "observed_rate": sum(y for _, y in g) / len(g)} for g in groups if g]


def betting_roi(outcomes: Sequence[str], decimal_odds: Sequence[float]) -> float:
    """Unit-stake ROI; outcomes are WIN, LOSS, or PUSH, with push returning stake."""
    if not outcomes or len(outcomes) != len(decimal_odds):
        raise ValueError("outcomes and decimal odds must be non-empty and the same length")
    profit = 0.0
    for outcome, odds in zip(outcomes, decimal_odds):
        if odds <= 1:
            raise ValueError("decimal odds must be greater than 1")
        if outcome == "WIN":
            profit += odds - 1
        elif outcome == "LOSS":
            profit -= 1
        elif outcome != "PUSH":
            raise ValueError("outcomes must be WIN, LOSS, or PUSH")
    return profit / len(outcomes)


def bootstrap_mean_interval(values: Sequence[float], confidence: float = 0.95,
                            iterations: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """Percentile interval for the mean; callers must pass only held-out results."""
    if not values or not 0 < confidence < 1 or iterations < 1:
        raise ValueError("values, confidence in (0, 1), and positive iterations are required")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iterations))
    tail = (1 - confidence) / 2
    return means[int(tail * iterations)], means[min(iterations - 1, int((1 - tail) * iterations))]
