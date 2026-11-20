import numpy as np
import pytest

from MiBici.modeling.train import calculate_metrics


def test_metrics_are_zero_for_exact_predictions():
    metrics = calculate_metrics([2, 4, 6], [2, 4, 6])

    assert metrics == {"mae": 0.0, "bias": 0.0}


def test_positive_bias_means_overestimation():
    metrics = calculate_metrics([2, 4], [3, 6])

    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["bias"] == pytest.approx(1.5)


@pytest.mark.parametrize(
    "actual,predictions",
    [([1, 2], [1]), ([], []), ([1, np.nan], [1, 2])],
)
def test_metrics_reject_invalid_inputs(actual, predictions):
    with pytest.raises(ValueError):
        calculate_metrics(actual, predictions)
