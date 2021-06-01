from pathlib import Path

import torch
from sacred import Ingredient
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn

ROOT = Path(__file__).parent
global_ingredient = Ingredient("g", save_git_info=False)     # global
device_ingredient = Ingredient("d", save_git_info=False)     # device


@global_ingredient.config
def global_arguments():
    """ Global Arguments """
    model_dir = "model_dir"     # str/null, Directory to save model parameters, graph, etc
    fileStorage = False         # bool, fileStorage observer
    mongodb = True              # bool, MongoDB observer
    mongo_port = 7000           # int, MongoDB port


@global_ingredient.config_hook
def global_hook(config, command_name, logger):
    g = config["g"]
    model_dir = g["model_dir"]
    if not model_dir:
        model_dir = "model_dir"
    model_dir = ROOT / model_dir
    if not model_dir.exists():
        model_dir.mkdir(exist_ok=True, parents=True)
    g["model_dir"] = str(model_dir)

    if g["mongodb"]:
        g["fileStorage"] = False
    else:
        g["fileStorage"] = True
    return g


@device_ingredient.config
def device_arguments():
    """ Device Arguments """
    enable_gpu = True               # Use gpu or not
    num_threads = 1                 #
    cudnn = {
        "enabled": True,
        "benchmark": True,          # Set True for better performance when the input size is fixed.
    }


@device_ingredient.config_hook
def device_hook(config, command_name, logger):
    d = config["d"]
    if d["enable_gpu"]:
        cudnn.enabled = d["cudnn"]["enabled"]
        cudnn.benchmark = d["cudnn"]["benchmark"]
        torch.set_num_threads(d["num_threads"])
    else:
        d["cudnn"]["enabled"] = False
        d["cudnn"]["benchmark"] = False
        d["num_threads"] = 0

    return d


def experiments_setup(ex):
    # Track outputs
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    # Track more source file
    ex.add_source_file("utils/timer.py")
    ex.add_source_file("core/losses.py")
    ex.add_source_file("core/metrics.py")

    # Add dependencies
    ex.add_package_dependency("opencv", "3.4.2")
    ex.add_package_dependency("cudatoolkit", "10.1.243")
    ex.add_package_dependency("cudnn", "7.6.5")
