# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to compile a Rainforests calibration model."""

from pathlib import Path

from improver import cli


@cli.clizefy
def process(
    lightgbm_model: cli.inputpath,
    output_dir: cli.inputpath,
    toolchain: str = "gcc",
    verbose: bool = False,
    parallel_comp: int = 0
):
    """Compile Rainforests LightGBM booster into a Treelite predictors.

    Args:
        lightgbm_model (LightGBM Booster file):
            Text file containing a representation of a LightGBM Booster.
        output_dir (directory):
            Directory where compiled Treelite predictor file should be created.
        toolchain (str):
            Toolchain to use for Treelite model compilation.
            'gcc' (default), 'msvc', 'clang' or a specific variation of clang or gcc
            (e.g. 'gcc-7').
        verbose (bool):
            Print verbose output
        parallel_comp (int):
            Enables parallel compilation to improve compilation time and reduce memory
            consumption during compilation.
            Defaults to 0 (no parallel compilation)
    """

    from improver.calibration.rainforest_compiler import RainforestsCompiler

    if not Path.is_file(lightgbm_model):
        raise ValueError("--lightgbm_model must be an existing file")
    if not Path.is_dir(output_dir):
        raise ValueError("--output_dir must be an existing directory")

    input_path = Path(lightgbm_model)
    output_dir = Path(output_dir)

    plugin = RainforestsCompiler(toolchain, verbose, parallel_comp)
    plugin.process(input_path, output_dir)
