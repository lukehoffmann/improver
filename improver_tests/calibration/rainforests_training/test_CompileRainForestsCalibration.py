# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import pytest

from improver.calibration.rainforest_training import (
    CompileRainForestsCalibration,
)


@pytest.mark.parametrize("available", [True, False])
def test__init__(treelite_available, tmp_path):
    """Test class is created if treelight libraries are available."""
    """Test class is not created if treelight libraries not available."""

    if treelite_available:
        expected_class = "CompileRainForestsCalibration"
        result = CompileRainForestsCalibration()
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = CompileRainForestsCalibration()


def test_process(dummy_lightgbm_models, tmp_path):
    """Test models are compiled."""

    tree_models, lead_times, thresholds = dummy_lightgbm_models

    compiler = CompileRainForestsCalibration()

    model = tree_models[lead_times[0], thresholds[0]]

    result = compiler.process(model, tmp_path)

    # assert isinstance(result, Dict)
    # for t in thresholds:
    #     assert isinstance(result[t], lightgbm.Booster)
