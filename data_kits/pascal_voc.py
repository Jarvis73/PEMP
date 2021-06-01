import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

cache_image = {}
cache_label = {}
cv_split = [[1,  2,  3,  4,  5],
            [6,  7,  8,  9,  10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20]]
class_names = [
    ["aeroplane",    "bicycle", "bird",  "boat",      "bottle"],
    ["bus",          "car",     "cat",   "chair",     "cow"],
    ["diningtable",  "dog",     "horse", "motorbike", "person"],
    ["potted plant", "sheep",   "sofa",  "train",     "tv/monitor"]
]



def crop_obj(image, mask, height, width, cls, name):
    margin_y = random.randint(0, mask.shape[0] - height)
    margin_x = random.randint(0, mask.shape[1] - width)
    mask_patch = mask[margin_y:margin_y + height, margin_x:margin_x + width]

    if np.count_nonzero(mask_patch) < 1024:  # For small foregrounds
        y_ = np.where(np.asarray(mask).max(axis=1) > 0)[0]
        x_ = np.where(np.asarray(mask).max(axis=0) > 0)[0]
        ymin, ymax = np.min(y_), np.max(y_) + 1
        xmin, xmax = np.min(x_), np.max(x_) + 1
        crop_y_range_start = max(0, ymax - height)
        crop_y_range_stop = max(min(mask.shape[0] - height, ymin), crop_y_range_start)
        crop_x_range_start = max(0, xmax - width)
        crop_x_range_stop = max(min(mask.shape[1] - width, xmin), crop_x_range_start)
        margin_y = random.randint(crop_y_range_start, crop_y_range_stop)
        margin_x = random.randint(crop_x_range_start, crop_x_range_stop)
        mask_patch = mask[margin_y:margin_y + height, margin_x:margin_x + width]
        # DEBUG
        if np.count_nonzero(mask_patch) == 0:
            number_try = 0
            while True:
                margin_y = random.randint(0, mask.shape[0] - height)
                margin_x = random.randint(0, mask.shape[1] - width)
                mask_patch = mask[margin_y:margin_y + height, margin_x:margin_x + width]
                if np.count_nonzero(mask_patch) > 0:
                    break
                number_try += 1
                if number_try > 100:
                    break
            if number_try > 100:
                print("Warning: full-zero mask")
    elif np.count_nonzero(255 - mask_patch) < 1024:    # For small backgrounds
        y_ = np.where(np.asarray(255 - mask).max(axis=1) > 0)[0]
        x_ = np.where(np.asarray(255 - mask).max(axis=0) > 0)[0]
        ymin, ymax = np.min(y_), np.max(y_) + 1
        xmin, xmax = np.min(x_), np.max(x_) + 1
        crop_y_range_start = max(0, ymax - height)
        crop_y_range_stop = max(min(mask.shape[0] - height, ymin), crop_y_range_start)
        crop_x_range_start = max(0, xmax - width)
        crop_x_range_stop = max(min(mask.shape[1] - width, xmin), crop_x_range_start)
        margin_y = random.randint(crop_y_range_start, crop_y_range_stop)
        margin_x = random.randint(crop_x_range_start, crop_x_range_stop)
        mask_patch = mask[margin_y:margin_y + height, margin_x:margin_x + width]
        # DEBUG
        if np.count_nonzero(255 - mask_patch) == 0:
            number_try = 0
            while True:
                margin_y = random.randint(0, mask.shape[0] - height)
                margin_x = random.randint(0, mask.shape[1] - width)
                mask_patch = mask[margin_y:margin_y + height, margin_x:margin_x + width]
                if np.count_nonzero(mask_patch) > 0:
                    break
                number_try += 1
                if number_try > 100:
                    break
            if number_try > 100:
                print("Warning: full-zero mask")
    image_patch = image[:, margin_y:margin_y + height, margin_x:margin_x + width]
    return image_patch, mask_patch


class PascalVOCTrain(Dataset):
    """
    PASCAL-5i dataset for training.

    Parameters
    ----------
    cfg: Config
    """
    def __init__(self, cfg, split, shot, query,
                 ret_name=False, one_cls=0):
        super(PascalVOCTrain, self).__init__()
        self.cfg = cfg
        self.train = True
        self.split = split
        self.shot = shot
        self.query = query
        self.base_dir = Path(self.cfg.base_dir)
        self.img_dir = self.base_dir / "JPEGImages"
        self.lab_dir = self.base_dir / "Binary_map_aug/train"
        self.id_dir = self.base_dir / "Binary_map_aug/train"
        self.ret_name = ret_name
        self.one_cls = one_cls
        self.cache = self.cfg.cache

        self.init()
        self.reset_sampler()

    @property
    def classes(self):
        return list(set(range(1, 21)) - set(range(self.split * 5 + 1, self.split * 5 + 6)))

    def reset_sampler(self):
        seed = self.cfg.seed
        test_seed = self.cfg.test_seed
        # Use fixed test sampler(cfg.test_seed) for reproducibility
        self.sampler = np.random.RandomState(seed) if self.train else np.random.RandomState(test_seed)

    def sample_tasks(self):
        self.tasks = []
        if self.one_cls <= 0:
            for i in range(len(self)):
                cls = self.sampler.choice(self.classes)
                indices = self.sampler.choice(self.idx_by_class[cls], size=self.shot + self.query, replace=False)
                self.tasks.append((cls, [self.sample_by_class[cls][j] for j in indices]))
        else:
            cls = self.one_cls
            for i in range(len(self)):
                indices = self.sampler.choice(self.idx_by_class[cls], size=self.shot + self.query, replace=False)
                self.tasks.append((cls, [self.sample_by_class[cls][j] for j in indices]))

    def init(self):
        mean = self.cfg.mean
        std = self.cfg.std

        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        self.flip = lambda flag, x: x if flag < 0.5 else F.hflip(x)
        self.resize_image = lambda size, x: F.resize(x, size, interpolation=Image.BILINEAR)
        self.resize_mask = lambda size, x: F.resize(x, size, interpolation=Image.NEAREST)
        self.jitter = lambda x: torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)(x)

        self.idx_by_class = {}
        self.sample_by_class = {}
        for i in self.classes:
            self.sample_by_class[i] = (self.id_dir / f"{i}.txt").read_text().strip().splitlines()
            # Count remaining number
            self.idx_by_class[i] = len(self.sample_by_class[i])

    def __len__(self):
        return self.cfg.train_n

    def get_image(self, name, cache=True):
        if self.cache and cache:
            if name not in cache_image:
                cache_image[name] = Image.open(self.img_dir / f"{name}.jpg")
            return cache_image[name]
        else:
            return Image.open(self.img_dir / f"{name}.jpg")

    def get_label(self, cls, name, cache=True, new_label=False):
        # new_label is not used in this function
        _ = new_label

        if self.cache and cache:
            blend_name = f"{cls}_{name}"
            if blend_name not in cache_label:
                cache_label[blend_name] = Image.open(self.lab_dir / f"{cls}/{name}.png")
            return cache_label[blend_name]
        else:
            return Image.open(self.lab_dir / f"{cls}/{name}.png")

    def __getitem__(self, idx):
        cls, all_names = self.tasks[idx]
        sup_names = all_names[:self.shot]  # len = self.shot
        qry_names = all_names[self.shot:]
        return self._get_episode(cls, sup_names, qry_names)

    def _get_episode(self, cls, sup_names, qry_names):
        height = self.cfg.height
        width = self.cfg.width

        # preprocess
        sup_rgbs, sup_masks = [], []
        for sup_name in sup_names:
            if self.train:
                scaled_factor = random.uniform(1, 1.5)
                scaled_size = (int(height * scaled_factor), int(width * scaled_factor))
                flag = random.random()
                sup_rgb = self.normalize(self.to_tensor(self.flip(flag, self.jitter(self.resize_image(
                    scaled_size, self.get_image(sup_name))))))
                sup_mask_a = np.array(self.flip(flag, self.resize_mask(
                    scaled_size, self.get_label(cls, sup_name))), np.uint8)
            else:
                scaled_size = (height, width)
                sup_rgb = self.normalize(self.to_tensor(self.resize_image(
                    scaled_size, self.get_image(sup_name))))
                sup_mask_a = np.array(self.resize_mask(
                    scaled_size, self.get_label(cls, sup_name)), np.uint8)

            if self.train:
                sup_rgb, sup_mask_a = crop_obj(sup_rgb, sup_mask_a, height, width, cls, sup_name)

            sup_mask_t = torch.from_numpy((sup_mask_a // 255).astype(np.float32))
            sup_mask_t = torch.stack((sup_mask_t, 1 - sup_mask_t), dim=0)
            sup_rgbs.append(sup_rgb)
            sup_masks.append(sup_mask_t)
        sup_rgb = torch.stack(sup_rgbs, dim=0)                          # [shot, 3, height, width]
        sup_mask = torch.stack(sup_masks, dim=0)                       # [shot, 2, height, width]

        qry_rgbs, qry_masks = [], []
        for qry_name in qry_names:
            if self.train:
                scaled_size = (height, width)
                flag = random.random()
                qry_rgb = self.normalize(self.to_tensor(self.flip(flag, self.jitter(self.resize_image(
                    scaled_size, self.get_image(qry_name))))))
                qry_mask_a = np.array(self.flip(flag, self.resize_mask(
                    scaled_size, self.get_label(cls, qry_name))), np.uint8)
            else:
                scaled_size = (height, width)
                qry_rgb = self.normalize(self.to_tensor(self.resize_image(
                    scaled_size, self.get_image(qry_name))))
                qry_mask_a = np.array(self.get_label(cls, qry_name), np.uint8)

            qry_mask_t = torch.from_numpy((qry_mask_a // 255).astype(np.int64))

            qry_rgbs.append(qry_rgb)
            qry_masks.append(qry_mask_t)
        qry_rgb = torch.stack(qry_rgbs, dim=0)  # [query, 3, height, width]
        qry_mask = torch.stack(qry_masks, dim=0)  # [query, height, width] / [query, H, W]

        if self.ret_name:
            return (sup_rgb, sup_mask, qry_rgb), qry_mask, cls, sup_names, qry_names
        return (sup_rgb, sup_mask, qry_rgb), qry_mask, cls


class PascalVOCTest(PascalVOCTrain, Dataset):
    """ divide the dataset into training set and test set:
        training set: 4 splits, each 3 for training
        test set: 4 splits, each one for testing
    """

    def __init__(self,  cfg, split, shot, query,
                 ret_name=False, one_cls=0):
        Dataset.__init__(self)
        self.cfg = cfg
        self.train = False
        self.split = split
        self.shot = shot
        self.query = query
        self.base_dir = Path(self.cfg.base_dir)
        self.img_dir = self.base_dir / "JPEGImages"
        self.lab_dir = self.base_dir / "Binary_map_aug/val"
        self.id_dir = self.base_dir / "Binary_map_aug/val"
        self.ret_name = ret_name
        self.one_cls = one_cls
        self.cache = self.cfg.cache

        self.init()
        self.reset_sampler()

    @property
    def classes(self):
        return list(range(self.split * 5 + 1, self.split * 5 + 6))

    def __len__(self):
        return self.cfg.test_n


class PascalVOCTrainCaNet(Dataset):
    """ CaNet dataloader

    Parameters
    ----------
    cfg: misc.MapConfig
        Experiment configuration
    split: int
        The split number
    shot: int
        The number of support images
    query: int
        The number of query images
    """
    def __init__(self, cfg, split, shot, query):
        super(PascalVOCTrainCaNet, self).__init__()
        self.cfg = cfg
        self.train = True
        self.split = split
        self.shot = shot
        self.query = query
        self.base_dir = Path(self.cfg.base_dir)
        self.img_dir = self.base_dir / "JPEGImages"
        self.lab_dir = self.base_dir / "Binary_map_aug/train"
        self.id_dir = self.base_dir / "Binary_map_aug/train"
        self.cache = self.cfg.cache

        self.init()
        self.reset_sampler()

    @property
    def classes(self):
        return list(set(range(1, 21)) - set(range(self.split * 5 + 1, self.split * 5 + 6)))

    def reset_sampler(self):
        seed = self.cfg.seed
        test_seed = self.cfg.test_seed
        # Use fixed test sampler(cfg.test_seed) for reproducibility
        self.sampler = np.random.RandomState(seed) \
            if self.train else np.random.RandomState(test_seed)
        self.history_sampler = np.random.RandomState(9876)

    def sample_tasks(self):
        self.tasks = []
        for i in range(len(self)):
            cls = self.sampler.choice(self.classes)
            indices = self.sampler.choice(self.idx_by_class[cls], size=self.shot + self.query, replace=False)
            self.tasks.append((cls, [self.sample_by_class[cls][j] for j in indices], indices[self.shot:]))
        self.history_mask_list = {c: [None] * self.idx_by_class[c] for c in self.classes}

    def init(self):
        mean = self.cfg.mean
        std = self.cfg.std
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        self.flip = lambda flag, x: x if flag < 0.5 else F.hflip(x)
        self.resize_image = lambda size, x: F.resize(x, size, interpolation=Image.BILINEAR)
        self.resize_mask = lambda size, x: F.resize(x, size, interpolation=Image.NEAREST)
        self.jitter = lambda x: torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)(x)

        self.idx_by_class = {}
        self.sample_by_class = {}
        for i in self.classes:
            self.sample_by_class[i] = (self.id_dir / f"{i}.txt").read_text().strip().splitlines()
            # Count remaining number
            self.idx_by_class[i] = len(self.sample_by_class[i])

    def __len__(self):
        return self.cfg.train_n

    def get_image(self, name):
        if self.cache:
            if name not in cache_image:
                cache_image[name] = Image.open(self.img_dir / f"{name}.jpg")
            return cache_image[name]
        else:
            return Image.open(self.img_dir / f"{name}.jpg")

    def get_label(self, cls, name):
        if self.cache:
            blend_name = f"{cls}_{name}"
            if blend_name not in cache_label:
                cache_label[blend_name] = Image.open(self.lab_dir / f"{cls}/{name}.png")
            return cache_label[blend_name]
        else:
            return Image.open(self.lab_dir / f"{cls}/{name}.png")

    def __getitem__(self, idx):
        height = self.cfg.height
        width = self.cfg.width
        cls, all_names, indices = self.tasks[idx]
        sup_names = all_names[:self.shot]  # len = self.shot
        qry_names = all_names[self.shot:]

        # preprocess
        sup_rgbs, sup_masks = [], []
        for sup_name in sup_names:
            if self.train:
                scaled_factor = random.uniform(1, 1.5)
                scaled_size = (int(height * scaled_factor), int(width * scaled_factor))
                flag = random.random()
                sup_rgb = self.normalize(self.to_tensor(self.flip(flag, self.jitter(self.resize_image(
                    scaled_size, self.get_image(sup_name))))))
                sup_mask_a = np.array(self.flip(flag, self.resize_mask(
                    scaled_size, self.get_label(cls, sup_name))), np.uint8)
            else:
                scaled_size = (height, width)
                sup_rgb = self.normalize(self.to_tensor(self.resize_image(
                    scaled_size, self.get_image(sup_name))))
                sup_mask_a = np.array(self.resize_mask(
                    scaled_size, self.get_label(cls, sup_name)), np.uint8)

            if self.train:
                sup_rgb, sup_mask_a = crop_obj(sup_rgb, sup_mask_a, height, width, cls, sup_name)

            sup_mask_t = torch.from_numpy((sup_mask_a // 255).astype(np.float32))
            sup_mask_t = torch.stack((sup_mask_t, 1 - sup_mask_t), dim=0)
            sup_rgbs.append(sup_rgb)
            sup_masks.append(sup_mask_t)
        sup_rgb = torch.stack(sup_rgbs, dim=0)                          # [shot, 3, height, width]
        sup_mask = torch.stack(sup_masks, dim=0)                       # [shot, 2, height, width]

        qry_rgbs, qry_masks = [], []
        for qry_name in qry_names:
            if self.train:
                scaled_size = (height, width)
                flag = random.random()
                qry_rgb = self.normalize(self.to_tensor(self.flip(flag, self.jitter(self.resize_image(
                    scaled_size, self.get_image(qry_name))))))
                qry_mask_a = np.array(self.flip(flag, self.resize_mask(
                    scaled_size, self.get_label(cls, qry_name))), np.uint8)
            else:
                scaled_size = (height, width)
                qry_rgb = self.normalize(self.to_tensor(self.resize_image(
                    scaled_size, self.get_image(qry_name))))
                qry_mask_a = np.array(self.get_label(cls, qry_name), np.uint8)

            qry_mask_t = torch.from_numpy((qry_mask_a // 255).astype(np.int64))

            qry_rgbs.append(qry_rgb)
            qry_masks.append(qry_mask_t)
        qry_rgb = torch.stack(qry_rgbs, dim=0)  # [query, 3, height, width]
        qry_mask = torch.stack(qry_masks, dim=0)  # [query, height, width] / [query, H, W]

        all_history_masks = []
        for index in indices:
            history_mask = self.history_mask_list[cls][index]
            if history_mask is None:
                history_mask = torch.zeros(2, (self.cfg.height - 1) // 8 + 1, (self.cfg.width - 1) // 8 + 1).fill_(0.)
            else:
                if self.train and self.history_sampler.random() <= 0.3:
                    history_mask = torch.zeros(2, (self.cfg.height - 1) // 8 + 1, (self.cfg.width) // 8 + 1).fill_(0.)
            all_history_masks.append(history_mask)
        history_mask = torch.stack(all_history_masks, dim=0)

        return (sup_rgb, sup_mask, qry_rgb), qry_mask, cls, indices, history_mask


class PascalVOCTestCaNet(PascalVOCTrainCaNet, Dataset):
    def __init__(self,  cfg, split, shot, query,
                 ret_name=False, one_cls=0):
        Dataset.__init__(self)
        self.cfg = cfg
        self.train = False
        self.split = split
        self.shot = shot
        self.query = query
        self.base_dir = Path(self.cfg.base_dir)
        self.img_dir = self.base_dir / "JPEGImages"
        self.lab_dir = self.base_dir / "Binary_map_aug/val"
        self.id_dir = self.base_dir / "Binary_map_aug/val"
        self.ret_name = ret_name
        self.one_cls = one_cls
        self.cache = self.cfg.cache

        self.init()
        self.reset_sampler()

    @property
    def classes(self):
        return list(range(self.split * 5 + 1, self.split * 5 + 6))

    def __len__(self):
        return self.cfg.test_n


def load(cfg, train_mode, split, shot, query,
         bs, test_bs, num_workers, pin_memory, one_cls,
         ret_name=False):
    """
    Load Pascal VOC 2012 dataset and return the batch generator.

    Parameters
    ----------
    cfg: misc.MapConfig
        Data configuration
    train_mode: str
        train/test mode
    split: int
    shot: int
    query: int
    bs, test_bs, num_workers, pin_memory, one_cls:
        [Config]
    ret_name: bool
        Yield sample names or not.

    Returns
    -------
    dataset: Dataset instance
    data_loader: torch.utils.data.DataLoader
        For each batch:
            sup_rgb: torch.Tensor,
                Support images of the shape [B, S, 3, H, W] and dtype float32
            sup_mask: torch.Tensor
                Support labels of the shape [B, S, 2, H, W] and dtype float32
            qry_rgb: torch.Tensor
                Query images of the shape [B, Q, 3, H, W] and dtype float32
            qry_mask: torch.Tensor(for train)
                Query labels of the shape [B, Q, H, W] or [bs, query, ori_H, ori_W] and dtype int64 (i.e. long)
            cls: list of int
                Index of the classes of the shape [bs]
            sup_names: list of str
                Names of support images
            qry_names: list of str
                Names of query images
    num_classes: int
        Number of classes
    """
    mode_error = "Not support training mode `{}`. " \
                 "Selected from [train, test, eval_online, train_canet, test_canet, test_canet_v2]"

    if "train" in train_mode:
        if train_mode == "train":
            dataset = PascalVOCTrain(cfg, split, shot, query, ret_name=ret_name, one_cls=one_cls)
        elif train_mode == "train_canet":
            dataset = PascalVOCTrainCaNet(cfg, split, shot, query)
        else:
            raise ValueError(mode_error.format(train_mode))

        data_loader = DataLoader(dataset,
                                 batch_size=bs,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=False)
    else:
        if train_mode in ["test", "eval_online"]:
            dataset = PascalVOCTest(cfg, split, shot, query, ret_name=ret_name, one_cls=one_cls)
        elif train_mode == "test_canet":
            dataset = PascalVOCTestCaNet(cfg, split, shot, query, ret_name=ret_name)
        else:
            raise ValueError(mode_error.format(train_mode))

        data_loader = DataLoader(dataset,
                                 batch_size=test_bs,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=False)

    num_classes = 20
    return dataset, data_loader, num_classes


class OneExampleLoader(PascalVOCTest):
    def __init__(self, cfg, split, shot, query):
        super(OneExampleLoader, self).__init__(cfg, split, shot, query)
        self.cache = False

    def reset_sampler(self):
        pass

    def sample_tasks(self):
        pass

    def load(self, cls, sup_names, qry_names):
        (sup_rgb, sup_mask, qry_rgb), qry_mask, cls = self._get_episode(cls, sup_names, qry_names)
        sup_rgb = sup_rgb.unsqueeze(dim=0)
        sup_mask = sup_mask.unsqueeze(dim=0)
        qry_rgb = qry_rgb.unsqueeze(dim=0)
        qry_mask = qry_mask.unsqueeze(dim=0)
        cls = [cls]
        return (sup_rgb, sup_mask, qry_rgb), qry_mask, cls
