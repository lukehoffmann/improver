import sys

import pytest
import numpy as np

from improver.calibration import (
    treelite_packages_available,
    lightgbm_package_available,
)

from ..rainforests_calibration.conftest import (
    lead_times,
    thresholds,
    generate_aligned_feature_cubes,
    generate_forecast_cubes,
    deterministic_features,
    deterministic_forecast,
    ensemble_features,
    ensemble_forecast,
    prepare_dummy_training_data,
)

@pytest.fixture
def treelite_available(available, monkeypatch):
    available = available and treelite_packages_available()
    if not available:
        monkeypatch.setitem(sys.modules, "treelite", None)
    return available


@pytest.fixture
def deterministic_training_data(
    deterministic_features, deterministic_forecast, lead_times
    ):
    return prepare_dummy_training_data(
        deterministic_features, deterministic_forecast, lead_times
    )
