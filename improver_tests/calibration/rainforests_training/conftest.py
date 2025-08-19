# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import sys

import pytest

from improver.calibration import treelite_packages_available

from ..rainforests_calibration.conftest import (
    deterministic_features,
    deterministic_forecast,
    ensemble_features,
    ensemble_forecast,
    lead_times,
    prepare_dummy_training_data,
    thresholds,
)

_ = (
    lead_times,
    thresholds,
    deterministic_features,
    deterministic_forecast,
    ensemble_features,
    ensemble_forecast,
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
