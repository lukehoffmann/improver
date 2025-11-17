# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

# from pathlib import Path

from improver import BasePlugin
from improver.calibration import (
    treelite_packages_available,
)

LIGHTGBM_EXTENSION = ".txt"
TREELITE_EXTENSION = ".so"


class RainforestsCompiler(BasePlugin):
    def __init__(self, toolchain="gcc", verbose=False, parallel_comp=0):
        self.treelite_available = treelite_packages_available()
        if not self.treelite_available:
            raise ModuleNotFoundError("Could not find TreeLite module")

        self.toolchain = toolchain
        self.verbose = verbose
        self.treelight_params = {"parallel_comp": parallel_comp, "quantize": 1}


    def process(self, lightgbm_model_file, output_dir):
        """Compile a lightgbm model."""
        import tl2cgen
        import treelite

        # Input validation
        if lightgbm_model_file.suffix.lower() != LIGHTGBM_EXTENSION:
            raise ValueError(f"Input path must have the extension {LIGHTGBM_EXTENSION}")
        if not output_dir.is_dir():
            raise ValueError("Output path must be a directory")

        output_filepath = output_dir / f"{lightgbm_model_file.stem}{TREELITE_EXTENSION}"

        model = treelite.frontend.load_lightgbm_model(lightgbm_model_file)

        tl2cgen.export_lib(
            model,
            toolchain=self.toolchain,
            libpath=output_filepath,
            verbose=self.verbose,
            params=self.treelight_params,
        )
