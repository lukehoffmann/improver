

# class TrainRainforestsModel(PostProcessingPlugin):
#     def __new__()

import lightgbm
import treelite
import treelite_runtime


lightgbm_params = {"objective": "binary", "num_leaves": 5, "verbose": -1, "seed": 0}

treelight_params = {"parallel_comp": 8, "quantize": 1}


def train_models(training_data, obs_column, train_columns, lead_times, thresholds):
    """Train a model for each threshold."""

    return {
        (lead_time, threshold): 
            train_model(training_data, obs_column, train_columns, lead_time, threshold)
        for lead_time in lead_times
        for threshold in thresholds
    }


def train_model(training_data, obs_column, train_columns, lead_time, threshold):
    """Train a model for each threshold."""

    curr_training_data = training_data.loc[
        training_data["lead_time_hours"] == lead_time
    ]
    data = lightgbm.Dataset(
        curr_training_data[train_columns],
        label=(curr_training_data[obs_column] >= threshold).astype(int),
    )
    booster = lightgbm.train(lightgbm_params, data, num_boost_round=10)
    return booster


def compile_models(lightgbm_models, lead_times, thresholds, path):
    """Compile lightgbm models."""

    return {
        (lead_time, threshold):
            compile_model(lightgbm_models[lead_time, threshold], path)
        for lead_time in lead_times
        for threshold in thresholds
    }


def compile_model(lightgbm_model, path):
    """Compile a lightgbm model."""

    libpath = str(path / "model.so")
    treelite_model = treelite.Model.from_lightgbm(lightgbm_model)
    treelite_model.export_lib(
        toolchain="gcc",
        libpath=libpath,
        verbose=False,
        params=treelight_params,
    )
    predictor = treelite_runtime.Predictor(
        libpath, verbose=True, nthread=1
    )
    return predictor

