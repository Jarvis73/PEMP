from pathlib import Path

from sacred import Ingredient

from data_kits import coco
from data_kits import pascal_voc

data_ingredient = Ingredient("data", save_git_info=False)
supported_datasets = ["PASCAL", "COCO"]     # PSACAL VOC 2012 and MS COCO 2014
error_info = f"Dataset not found. Please Select from {supported_datasets}."


@data_ingredient.config
def data_config():
    """ Dataset configuration """
    dataset = "PASCAL"              # str, dataset name. [PASCAL, COCO]
    base_dir = ""                   # str, dataset directory
    mean = [0.485, 0.456, 0.406]    # list, normalization mean in data preprocessing
    std = [0.229, 0.224, 0.225]     # list, normalization std in data preprocessing
    height = 401                    # int, input image height
    width = 401                     # int, input image width
    bs = 4                          # int, batch size
    test_bs = 1                     # int, test batch size (Don't change it!)
    num_workers = min(bs, 4)        # int, PyTorch DataLoader argument
    pin_memory = True               # bool, PyTorch DataLoader argument
    train_n = 5000                  # int, number of train examples in each epoch (for balancing dataset)
    test_n = 1000                   # int, number of test examples in each run
    seed = 1234                     # int, training set random seed
    test_seed = 5678                # int, test set random seed
    one_cls = 0                     # int, load dataset of a specified class. PASCAL: 1-20, COCO: 1-80
    cache = True                    # bool, cache datasets during training and testing


@data_ingredient.config_hook
def data_hook(config, command_name, logger):
    data = config["data"]
    if data["dataset"] == "PASCAL":
        base_dir = Path(__file__).parent.parent / "data/VOCdevkit/VOC2012"
        if not base_dir.exists():
            raise FileNotFoundError(f"Dataset PASCAL is not found in {str(base_dir)}")
        data["base_dir"] = str(base_dir)
    elif data["dataset"] == "COCO":
        base_dir = Path(__file__).parent.parent / "data/COCO"
        if not base_dir.exists():
            raise FileNotFoundError(f"Dataset COCO is not found in {str(base_dir)}")
        data["base_dir"] = str(base_dir)
    else:
        raise ValueError(error_info)

    return data


@data_ingredient.capture
def load(cfg, train_mode, split, shot, query,
         dataset, bs, test_bs, num_workers, pin_memory, one_cls,
         ret_name=False, first=False, logger=None):

    if dataset == "PASCAL":
        loader = pascal_voc.load(cfg, train_mode, split, shot, query,
                                 bs, test_bs, num_workers, pin_memory, one_cls,
                                 ret_name=ret_name)
    elif dataset == "COCO":
        loader = coco.load(cfg, train_mode, split, shot, query,
                           bs, test_bs, num_workers, pin_memory,
                           ret_name=ret_name)
    else:
        raise ValueError

    if logger:
        logger.info(f"{'Initialize' if first else ' ' * 10} ==> Data loader {dataset} for {train_mode} mode")

    return loader


@data_ingredient.capture
def OneExampleLoader(cfg, split, shot, query):
    if cfg.dataset == "PASCAL":
        return pascal_voc.OneExampleLoader(cfg, split, shot, query)
    elif cfg.dataset == "COCO":
        return coco.OneExampleLoader(cfg, split, shot, query)


@data_ingredient.capture
def get_val_labels(split, dataset):
    """
    Get validation label list

    Parameters
    ----------
    split: int
        Split number
    dataset: str
        Dataset name. [PASCAL, COCO]

    Returns
    -------

    """
    if dataset == "PASCAL":
        return list(range(split * 5 + 1, split * 5 + 6))
    elif dataset == "COCO":
        return list(range(split * 20 + 1, split * 20 + 21))
    else:
        raise ValueError(error_info)


def get_class_name(cls, dataset):
    cls = int(cls) - 1

    if dataset == "PASCAL":
        name = pascal_voc.class_names[cls // 5][cls % 5]
    elif dataset == "COCO":
        name = coco.class_names[cls // 20][cls % 20]
    else:
        raise ValueError(error_info)

    return "_".join(name.replace("/", " ").split(" "))
