

import pytest

from improver.calibration.rainforest_training import TrainRainForestsCalibration

@pytest.mark.parametrize("available", [True, False])
def test__new__(treelite_available):
    """Test treelight class is created if treelight libraries are available."""

    if treelite_available:
        expected_class = "TrainRainForestsCalibrationTreelite"
    else:
        expected_class = "TrainRainForestsCalibrationLightGBM"

    result = TrainRainForestsCalibration()
    assert type(result).__name__ == expected_class
