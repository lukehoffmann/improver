# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from improver.calibration import (
    lightgbm_package_available,
    treelite_packages_available,
)


class TrainRainForestsCalibration:
    lightgbm_params = {
        "objective": "binary",
        "num_leaves": 5,
        "num_boost_round": 10,
        "verbose": -1,
        "seed": 0,
    }

    def __init__(self, thresholds, obs_column, train_columns):
        self.lightgbm_available = lightgbm_package_available()
        if not self.lightgbm_available:
            raise ModuleNotFoundError("Could not find LightGBM module")

        self.thresholds = list(float(t) for t in thresholds)
        self.obs_column = obs_column
        self.train_columns = train_columns

    def process(self, merged_data):
        """Train a model for each threshold."""

        return {t: self._train_lightgbm_model(t, merged_data) for t in self.thresholds}

    def _train_lightgbm_model(self, threshold, merged_data):
        """Train a model for one threshold."""
        import lightgbm

        threshold_met = (merged_data[self.obs_column] >= threshold).astype(int)
        training_data = merged_data[self.train_columns]
        dataset = lightgbm.Dataset(training_data, label=threshold_met)

        return lightgbm.train(self.lightgbm_params, dataset)


class CompileRainForestsCalibration:
    treelight_params = {"parallel_comp": 8, "quantize": 1}

    def __init__(self):
        self.treelite_available = treelite_packages_available()
        if not self.treelite_available:
            raise ModuleNotFoundError("Could not find TreeLite module")

    def process(self, lightgbm_model, output_filepath):
        """Compile a lightgbm model."""
        import tl2cgen
        import treelite

        model = treelite.Model.from_lightgbm(lightgbm_model)
        tl2cgen.export_lib(
            model,
            toolchain="gcc",
            libpath=output_filepath,
            verbose=False,
            params=self.treelight_params,
        )
