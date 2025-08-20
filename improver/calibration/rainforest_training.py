# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import os
from pathlib import PosixPath

from improver.calibration import (
    lightgbm_package_available,
    treelite_packages_available,
)


class TrainRainForestsCalibration:
    def __init__(self, lead_times, thresholds, obs_column, train_columns, output_dir):
        self.lightgbm_available = lightgbm_package_available()
        self.treelite_available = treelite_packages_available()

        if not self.lightgbm_available:
            raise ModuleNotFoundError("Could not find LightGBM module")

        self.lead_times = list(int(l) for l in lead_times)
        self.thresholds = list(float(t) for t in thresholds)
        self.obs_column = obs_column
        self.train_columns = train_columns
        self.output_dir = PosixPath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.config = {
            self._lead_time_key(l): {
                self._threshold_key(t): {} for t in self.thresholds
            }
            for l in self.lead_times
        }

    def process(self, training_data):
        """Train a model for each threshold."""
        for lead_time in self.lead_times:
            for threshold in self.thresholds:
                self.config[self._lead_time_key(lead_time)][
                    self._threshold_key(threshold)
                ] = self._train_model(lead_time, threshold, training_data)
        return self.config

    def _train_model(self, lead_time, threshold, training_data):
        result = {}
        filepath = self._model_filename(lead_time, threshold, "txt")
        lightgbm_model = self._train_lightgbm_model(lead_time, threshold, training_data)
        lightgbm_model.save_model(filepath)
        result["lightgbm_model"] = str(filepath)

        if self.treelite_available:
            filepath = self._model_filename(lead_time, threshold, "so")
            self._compile_treelite_model(lightgbm_model, filepath)
            result["treelite_model"] = str(filepath)
        return result

    def _lead_time_key(self, lead_time):
        return f"{lead_time:02d}"

    def _threshold_key(self, threshold):
        return f"{threshold:06.4f}"

    def _model_filename(self, lead_time, threshold, extension):
        return self.output_dir / f"{lead_time:03d}H_{threshold:06.4f}.{extension}"

    lightgbm_params = {
        "objective": "binary",
        "num_leaves": 5,
        "num_boost_round": 10,
        "verbose": -1,
        "seed": 0,
    }

    def _train_lightgbm_model(self, lead_time, threshold, training_data):
        """Train a model for each threshold."""
        import lightgbm

        curr_training_data = training_data.loc[
            training_data["lead_time_hours"] == lead_time
        ]
        label = (curr_training_data[self.obs_column] >= threshold).astype(int)
        data = lightgbm.Dataset(
            curr_training_data[self.train_columns],
            label=label,
        )

        return lightgbm.train(self.lightgbm_params, data)

    treelight_params = {"parallel_comp": 8, "quantize": 1}

    def _compile_treelite_model(self, lightgbm_model, filepath):
        """Compile a lightgbm model."""
        import tl2cgen
        import treelite

        model = treelite.Model.from_lightgbm(lightgbm_model)
        tl2cgen.export_lib(
            model,
            toolchain="gcc",
            libpath=filepath,
            verbose=False,
            params=self.treelight_params,
        )
        return tl2cgen.Predictor(filepath, verbose=True, nthread=1)
