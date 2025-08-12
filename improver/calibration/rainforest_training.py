

class TrainRainForestsCalibration():
    # pass
    def __new__(cls):
        """Initialise class object based on package availability."""
        try:
            # Use treelite class, unless subsequent conditions fail.
            cls = TrainRainForestsCalibrationTreelite
            # Try and initialise the tl2cgen library to test if the package
            # is available.
            import tl2cgen  # noqa: F401

        except (ModuleNotFoundError, ValueError):
            # Default to lightGBM.
            cls = TrainRainForestsCalibrationLightGBM

        return super(TrainRainForestsCalibration, cls).__new__(cls)

class TrainRainForestsCalibrationLightGBM(TrainRainForestsCalibration):

    lightgbm_params = {"objective": "binary", "num_leaves": 5, "verbose": -1, "seed": 0}

    def __new__(cls):
        return super(TrainRainForestsCalibration, cls).__new__(cls)

    def train_models(self, training_data, obs_column, train_columns, lead_times, thresholds):
        """Train a model for each threshold."""

        return {
            (lead_time, threshold): 
                self.train_model(training_data, obs_column, train_columns, lead_time, threshold)
            for lead_time in lead_times
            for threshold in thresholds
        }


    def train_model(self, training_data, obs_column, train_columns, lead_time, threshold):
        """Train a model for each threshold."""
        import lightgbm

        curr_training_data = training_data.loc[
            training_data["lead_time_hours"] == lead_time
        ]
        data = lightgbm.Dataset(
            curr_training_data[train_columns],
            label=(curr_training_data[obs_column] >= threshold).astype(int),
        )
        booster = lightgbm.train(self.lightgbm_params, data, num_boost_round=10)
        return booster

class TrainRainForestsCalibrationTreelite(TrainRainForestsCalibrationLightGBM):

    treelight_params = {"parallel_comp": 8, "quantize": 1}

    def __new__(cls):
        return super(TrainRainForestsCalibration, cls).__new__(cls)

    def compile_models(self, lightgbm_models, lead_times, thresholds, path):
        """Compile lightgbm models."""

        return {
            (lead_time, threshold):
                self.compile_model(lightgbm_models[lead_time, threshold], path)
            for lead_time in lead_times
            for threshold in thresholds
        }


    def compile_model(self, lightgbm_model, path):
        """Compile a lightgbm model."""
        import tl2cgen
        import treelite

        libpath = str(path / "model.so")
        treelite_model = treelite.Model.from_lightgbm(lightgbm_model)
        tl2cgen.export_lib(
            treelite_model,
            toolchain="gcc",
            libpath=libpath,
            verbose=False,
            params=self.treelight_params,
        )
        predictor = tl2cgen.Predictor(
            libpath, verbose=True, nthread=1
        )
        return predictor

