import os

from improver.calibration import (
    lightgbm_package_available,
    treelite_packages_available,
)


class TrainRainForestsCalibration:
    # pass
    def __new__(cls, lead_times, thresholds, output_path):
        """Initialise class object based on package availability."""
        treelite_available = treelite_packages_available()
        lightgbm_available = lightgbm_package_available()
        if not lightgbm_available:
            raise ModuleNotFoundError("Could not find tLightGBM module")

        if treelite_available:
            cls = TrainRainForestsCalibrationTreelite
        else:
            cls = TrainRainForestsCalibrationLightGBM
        return super(TrainRainForestsCalibration, cls).__new__(cls)

    def __init__(self, lead_times, thresholds, output_path):
        self.lead_times = list(int(l) for l in lead_times)
        self.thresholds = list(float(t) for t in thresholds)
        self.output_path = output_path

        self.config = {
            self.lead_time_key(l): {self.threshold_key(t): {} for t in self.thresholds}
            for l in self.lead_times
        }

    def process(self, training_data, obs_column, train_columns) -> None:
        """Subclasses should override this function."""
        raise NotImplementedError(
            "Process function must be called via subclass method."
        )
        return self.config

    def lead_time_key(self, lead_time):
        return f"{lead_time:02d}"

    def threshold_key(self, threshold):
        return f"{threshold:06.4f}"

    def model_filename(self, lead_time, threshold, extension):
        return f"{lead_time:03d}H_{threshold:06.4f}.{extension}"


class TrainRainForestsCalibrationLightGBM(TrainRainForestsCalibration):
    def __new__(cls, lead_times, thresholds, output_path):
        return super(TrainRainForestsCalibration, cls).__new__(cls)

    lightgbm_params = {
        "objective": "binary",
        "num_leaves": 5,
        "num_boost_round": 10,
        "verbose": -1,
        "seed": 0,
    }

    def process(self, training_data, obs_column, train_columns):
        """Train a model for each threshold."""
        output_path = self.output_path / "lightgbm_models"
        os.makedirs(output_path, exist_ok=True)

        for lead_time in self.lead_times:
            lkey = self.lead_time_key(lead_time)
            for threshold in self.thresholds:
                tkey = self.threshold_key(threshold)

                filepath = output_path / self.model_filename(lead_time, threshold, "txt")
                model = self._train_model(
                    training_data, obs_column, train_columns, lead_time, threshold
                )
                model.save_model(filepath)
                self.config[lkey][tkey]["lightgbm_model"] = str(filepath)

        return self.config

    def _train_model(
        self, training_data, obs_column, train_columns, lead_time, threshold
    ):
        """Train a model for each threshold."""
        import lightgbm

        curr_training_data = training_data.loc[
            training_data["lead_time_hours"] == lead_time
        ]
        label = (curr_training_data[obs_column] >= threshold).astype(int)
        data = lightgbm.Dataset(
            curr_training_data[train_columns],
            label=label,
        )

        return lightgbm.train(self.lightgbm_params, data)


    def _get_model(self, lead_time, threshold):
        import lightgbm

        lkey = self.lead_time_key(lead_time)
        tkey = self.threshold_key(threshold)
        lightgbm_model_filepath = self.config[lkey][tkey]['lightgbm_model']
        return lightgbm.Booster(model_file=lightgbm_model_filepath)

class TrainRainForestsCalibrationTreelite(TrainRainForestsCalibrationLightGBM):
    treelight_params = {"parallel_comp": 8, "quantize": 1}

    def __new__(cls):
        return super(TrainRainForestsCalibration, cls).__new__(cls)

    def process(self, training_data, obs_column, train_columns):
        TrainRainForestsCalibrationLightGBM.process(
            self, training_data, obs_column, train_columns
        )
        import lightgbm

        output_path = self.output_path / "treelite_models"
        os.makedirs(output_path, exist_ok=True)

        for lead_time in self.lead_times:
            lkey = self.lead_time_key(lead_time)
            for threshold in self.thresholds:
                tkey = self.threshold_key(threshold)

                lightgbm_model = self._get_model(lead_time, threshold)
                path = output_path / self.model_filename(lead_time, threshold, "so")
                self._compile_model(lightgbm_model, path)
                self.config[lkey][tkey]["treelite_model"] = str(path)

        return self.config

    def _compile_model(self, lightgbm_model, filepath):
        """Compile a lightgbm model."""
        import tl2cgen
        import treelite

        treelite_model = treelite.Model.from_lightgbm(lightgbm_model)
        tl2cgen.export_lib(
            treelite_model,
            toolchain="gcc",
            libpath=filepath,
            verbose=False,
            params=self.treelight_params,
        )
        predictor = tl2cgen.Predictor(filepath, verbose=True, nthread=1)
        return predictor
