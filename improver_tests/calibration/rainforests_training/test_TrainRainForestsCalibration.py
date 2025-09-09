# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import Dict

import lightgbm
import pytest

from improver.calibration.rainforest_training import (
    TrainRainForestsCalibration,
)


@pytest.mark.parametrize("available", [True, False])
def test__init__lightgmb_available(lightgbm_available):
    """Test class is created if lightgbm library is available."""
    """Test class is not created if lightgbm library not available."""

    if lightgbm_available:
        expected_class = "TrainRainForestsCalibration"
        result = TrainRainForestsCalibration([], "", [])
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = TrainRainForestsCalibration([], "", [])


def test__init__():
    """Test class is created with lead times and thresholds."""

    result = TrainRainForestsCalibration(
        [0.1, 0.05, 0.01], "column1", ["column2", "column3", "column4"]
    )
    assert result.thresholds == [0.1, 0.0500, 0.0100]
    assert result.obs_column == "column1"
    assert result.train_columns == ["column2", "column3", "column4"]


def test_process(thresholds, deterministic_training_data):
    """Test lightgbm models are created."""

    training_data, fcst_column, obs_column, train_columns = deterministic_training_data

    trainer = TrainRainForestsCalibration(thresholds, obs_column, train_columns)

    lead_time = 24
    curr_training_data = training_data.loc[
        training_data["lead_time_hours"] == lead_time
    ]

    result = trainer.process(curr_training_data)

    # Config is initialised as a nested set of dictionaries
    # LightGBM paths are present
    assert isinstance(result, Dict)
    for t in thresholds:
        assert isinstance(result[t], lightgbm.Booster)
