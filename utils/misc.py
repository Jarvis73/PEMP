import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from sacred.config.custom_containers import ReadOnlyDict
from sacred.observers import FileStorageObserver, MongoObserver

from utils import loggers

ROOT = Path(__file__).parents[1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_snapshot_files(_run):
    if _run.observers:
        for obs in _run.observers:
            if isinstance(obs, FileStorageObserver):
                for source_file, _ in _run.experiment_info['sources']:
                    os.makedirs(os.path.dirname(f'{obs.dir}/source/{source_file}'), exist_ok=True)
                    obs.save_file(source_file, f'source/{source_file}')
                shutil.rmtree(f'{obs.basedir}/_sources')
                break


def add_observers(ex, config, fileStorage=True, MongoDB=True, db_name="default"):
    if fileStorage:
        observer_file = FileStorageObserver(config["g"]["model_dir"])
        ex.observers.append(observer_file)

    if MongoDB:
        try:
            observer_mongo = MongoObserver(url="localhost:{}".format(config["g"]["mongo_port"]), db_name=db_name)
            ex.observers.append(observer_mongo)
        except ModuleNotFoundError:
            # Ignore Mongo Observer
            pass


def post_hook(ex, NAME):
    @ex.config_hook
    def config_hook(config, command_name, logger):
        g = config["g"]
        if command_name in ["train", "test"]:
            if config["split"] == -1:
                raise ValueError("Argument `split` is required! For example: `split=0` ")

            add_observers(ex, config, fileStorage=g["fileStorage"], MongoDB=g["mongodb"], db_name=NAME)
            ex.logger = loggers.get_global_logger(name=NAME)
        return config


class MapConfig(ReadOnlyDict):
    """
    A wrapper for dict. This wrapper allow users to access dict value by `dot` operation.
    For example, you can access `cfg["split"]` by `cfg.split`, which makes the code more clear.

    Notice that the result object is a sacred.config.custom_containers.ReadOnlyDict, which is
    a read-only dict for preserving the configuration.

    Parameters
    ----------
    obj: ReadOnlyDict
        Configuration dict.
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = MapConfig(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(MapConfig, self).__init__(new_dict, **kwargs)


def get_pth(exp_dir, ckpt):
    ckpt_name = [ckpt, "bestckpt.pth", "ckpt.pth"]
    for name in ckpt_name:
        if name is None:
            continue
        ckpt_path = exp_dir / name
        if ckpt_path.exists() and ckpt_path.is_file():
            return ckpt_path
    return False


def try_possible_ckpt_names(base, exp_id, ckpt=None):
    if isinstance(exp_id, int) and exp_id >= 0:
        # Search running results from `base` directory (with specific tag)
        ckpt_path = get_pth(base / str(exp_id), ckpt)
        if ckpt_path:
            return ckpt_path, exp_id

        # Search running results from `model_dir` (without specific tag)
        for exp_dir in base.parent.glob("*/[0-9]*"):
            if exp_dir.is_dir() and int(exp_dir.name) == exp_id:
                ckpt_path = get_pth(exp_dir, ckpt)
                return ckpt_path, exp_id

    # Use ckpt directly
    ckpt_file = Path(ckpt)
    if ckpt_file.exists():
        return ckpt_file, 99999999

    return False, 99999999


def find_snapshot(cfg, exp_id=-1, ckpt=None):
    """ Find experiment checkpoint """
    if ckpt is None:
        ckpt = cfg.ckpt

    base = Path(cfg.g.model_dir) / str(cfg.tag)
    ckpt_path = try_possible_ckpt_names(base, exp_id, ckpt)
    if ckpt_path[0]:
        return ckpt_path

    # Guess the maximum exp_id as the desired exp_id
    all_exp_ids = [int(dir_name.name) for dir_name in base.glob("[0-9]*") if dir_name.is_dir()]
    if len(all_exp_ids) > 0:
        exp_id = max(all_exp_ids)
        ckpt_path = try_possible_ckpt_names(base, exp_id, ckpt)
        if ckpt_path[0]:
            return ckpt_path

    # Import readline module to improve the experience of input
    # noinspection PyUnresolvedReferences
    import readline

    while True:
        ckpt_path = Path(input("Cannot find checkpoints. Please input:"))
        if ckpt_path.exists():
            return ckpt_path, 99999999
