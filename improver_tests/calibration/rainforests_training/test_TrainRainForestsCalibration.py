

import pytest
from typing import Dict

from improver.calibration.rainforest_training import (
    TrainRainForestsCalibration,
    TrainRainForestsCalibrationLightGBM,
)

@pytest.mark.parametrize("available", [True, False])
def test__new__(treelite_available):
    """Test treelight class is created if treelight libraries are available."""

    if treelite_available:
        expected_class = "TrainRainForestsCalibrationTreelite"
    else:
        expected_class = "TrainRainForestsCalibrationLightGBM"

    result = TrainRainForestsCalibration([], [], "")
    assert type(result).__name__ == expected_class


@pytest.mark.parametrize("available", [True, False])
def test__init__(treelite_available, tmp_path):
    """Test class is created with lead times and thresholds."""


    result = TrainRainForestsCalibration([ 24, 12 ], [ 0.1, 0.05, 0.01 ], tmp_path)
    assert result.lead_times == [ "24", "12" ]
    assert result.thresholds == [ "0.1", "0.0500", "0.0100" ]


@pytest.mark.parametrize("available", [True, False])
def test__init__(treelite_available, tmp_path):
    """Test class is initialised with config structure based on lead times and thresholds."""

    result = TrainRainForestsCalibration([ 36, 24, 12 ], [ 0.1, 0.01 ], tmp_path)

    # Config is initialised as a nested set of dictionaries
    assert result.config == {
        "36": { "0.1000": {}, "0.0100": {} },
        "24": { "0.1000": {}, "0.0100": {} },
        "12": { "0.1000": {}, "0.0100": {} }
    }


@pytest.mark.parametrize("available", [True, False])
def test__init__config(treelite_available, tmp_path):
    """Test class is initialised with config structure based on lead times and thresholds."""

    result = TrainRainForestsCalibration([ 24, 12 ], [ 0.1, 0.01 ], tmp_path)

    # Config is initialised as a nested set of dictionaries
    assert result.config == {
        "24": { "0.1000": {}, "0.0100": {} },
        "12": { "0.1000": {}, "0.0100": {} }
    }


@pytest.mark.parametrize("available", [True, False])
def test_process_lightgbm(treelite_available, lead_times, thresholds, deterministic_training_data, tmp_path):
    """Test lightgbm models are created."""

    trainer = TrainRainForestsCalibration(lead_times, thresholds, tmp_path)

    training_data, fcst_column, obs_column, train_columns = deterministic_training_data
    result = trainer.process(training_data, obs_column, train_columns)

    # Config is initialised as a nested set of dictionaries
    # LightGBM paths are present
    assert isinstance(result, Dict)
    assert isinstance(result["24"], Dict)
    assert isinstance(result["24"]["0.0000"], Dict)
    assert result["24"]["0.0000"]["lightgbm_model"] == f"{tmp_path}/lightgbm_models/024H_0.0000.txt"
    assert isinstance(result["24"]["0.0001"], Dict)
    assert result["24"]["0.0001"]["lightgbm_model"] == f"{tmp_path}/lightgbm_models/024H_0.0001.txt"
    assert isinstance(result["24"]["0.0010"], Dict)
    assert result["24"]["0.0010"]["lightgbm_model"] == f"{tmp_path}/lightgbm_models/024H_0.0010.txt"
    assert isinstance(result["24"]["0.0100"], Dict)
    assert result["24"]["0.0100"]["lightgbm_model"] == f"{tmp_path}/lightgbm_models/024H_0.0100.txt"


@pytest.mark.parametrize("available", [ False ])
def test_process_treelite_unavailable(treelite_available, lead_times, thresholds, deterministic_training_data, tmp_path):
    """Test treelite models are not created when treelite is not available."""

    trainer = TrainRainForestsCalibration(lead_times, thresholds, tmp_path)
    assert type(trainer).__name__ == "TrainRainForestsCalibrationLightGBM"

    training_data, fcst_column, obs_column, train_columns = deterministic_training_data

    result = trainer.process(training_data, obs_column, train_columns)

    # Config is initialised as a nested set of dictionaries
    # LightGBM and Treelite paths are present
    assert result["24"]["0.0000"].get("treelite_model") == None
    assert result["24"]["0.0010"].get("treelite_model") == None
    assert result["24"]["0.0010"].get("treelite_model") == None
    assert result["24"]["0.0100"].get("treelite_model") == None


@pytest.mark.parametrize("available", [ True ])
def test_process_treelite_available(treelite_available, lead_times, thresholds, deterministic_training_data, tmp_path):
    """Test treelite models are created when treelite is available."""

    trainer = TrainRainForestsCalibration(lead_times, thresholds, tmp_path)
    assert type(trainer).__name__ == "TrainRainForestsCalibrationTreelite"

    training_data, fcst_column, obs_column, train_columns = deterministic_training_data

    result = trainer.process(training_data, obs_column, train_columns)

    # Config is initialised as a nested set of dictionaries
    # LightGBM and Treelite paths are present
    assert result["24"]["0.0000"]["treelite_model"] == f"{tmp_path}/treelite_models/024H_0.0000.so"
    assert result["24"]["0.0001"]["treelite_model"] == f"{tmp_path}/treelite_models/024H_0.0001.so"
    assert result["24"]["0.0010"]["treelite_model"] == f"{tmp_path}/treelite_models/024H_0.0010.so"
    assert result["24"]["0.0100"]["treelite_model"] == f"{tmp_path}/treelite_models/024H_0.0100.so"



