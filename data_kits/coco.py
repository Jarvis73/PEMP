import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import tqdm
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

cache_image = {}
cache_label = {}
cv_split = [[1, 5, 9, 14, 18, 22, 27, 33, 37, 41, 46, 50, 54, 58, 62, 67, 74, 78, 82, 87],
            [2, 6, 10, 15, 19, 23, 28, 34, 38, 42, 47, 51, 55, 59, 63, 70, 75, 79, 84, 88],
            [3, 7, 11, 16, 20, 24, 31, 35, 39, 43, 48, 52, 56, 60, 64, 72, 76, 80, 85, 89],
            [4, 8, 13, 17, 21, 25, 32, 36, 40, 44, 49, 53, 57, 61, 65, 73, 77, 81, 86, 90]]
class_names = [
    ["person",       "airplane",       "boat",          "parking meter", "dog",          "elephant",    "backpack",
     "suitcase",     "sports ball",    "skateboard",    "wine glass",    "spoon",        "sandwich",    "hot dog",
     "chair",        "dining table",   "mouse",         "microwave",     "refrigerator", "scissors"],
    ["bicycle",      "bus",            "traffic light", "bench",         "horse",        "bear",        "umbrella",
     "frisbee",      "kite",           "surfboard",     "cup",           "bowl",         "orange",      "pizza",
     "couch",        "toilet",         "remote",        "oven",          "book",         "teddy bear"],
    ["car",          "train",          "fire hydrant",  "bird",          "sheep",        "zebra",       "handbag",
     "skis",         "baseball bat",   "tennis racket", "fork",          "banana",       "broccoli",    "donut",
     "potted plant", "tv",             "keyboard",      "toaster",       "clock",        "hair drier"],
    ["motorcycle",   "truck",          "stop sign",     "cat",           "cow",          "giraffe",     "tie",
     "snowboard",    "baseball glove", "bottle",        "knife",         "apple",        "carrot",      "cake",
     "bed",          "laptop",         "cell phone",    "sink",          "vase",         "toothbrush"],
]

new_index = {}
for i in range(4):
    for j in range(20):
        new_index[cv_split[i][j]] = i * 20 + j + 1


def crop_obj(image, mask, height, width, cls, name):
    margin_y = random.randint(0, mask.shape[0] - height)
    margin_x = random.randint(0, mask.shape[1] - width)
    mask_patch = mask[margin_y:margin_y + height, margin_x:margin_x + width]

    if 1024 > np.count_nonzero(mask_patch):
        # For small foregrounds
        # some samples have no foregrounds (after resize)
        # eg: COCO sample 448280 with class 77(phone) after resize (450,450)
        # http://images.cocodataset.org/train2014/COCO_train2014_000000448280.jpg
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
    elif 1024 > np.count_nonzero(255 - mask_patch):
        # For small backgrounds
        # some samples have no backgrounds
        # eg: COCO sample 103936 with class 6(bus)
        # http://images.cocodataset.org/train2014/COCO_train2014_000000103936.jpg
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


class COCOTrain(Dataset):
    def __init__(self, cfg, split, shot, query,
                 ret_name=False):
        super(COCOTrain, self).__init__()
        self.cfg = cfg
        self.train = True
        self.split = split
        self.shot = shot
        self.query = query
        self.base_dir = Path(self.cfg.base_dir)
        self.img_dir = self.base_dir / "train2014"
        self.annotaion_dir = self.base_dir / "annotations"
        self.coco = COCO(self.annotaion_dir / "instances_train2014.json")
        self.check_mask_threshold = 16
        self.list = self.base_dir / ("train2014_list_%d.json" % self.check_mask_threshold)
        self.ret_name = ret_name

        self.init()
        self.reset_sampler()
        self.count = 0

    def generate_file_list(self, threshold=16):
        """
        Remove objects smaller than `threshould`

        Parameters
        ----------
        threshold: int
            Threshould to filter out small objects

        Returns
        -------

        """
        def check_mask(cls, imgId):
            # Set cache to False to avoid out of memory
            label = np.array(self.get_label(cls, imgId, cache=False))
            if np.count_nonzero(255 - label) < threshold:
                # two small background
                return False
            elif np.count_nonzero(label) < threshold:
                # too small foreground
                return False
            else:
                return True

        print("No sample List Found. Generating now...")
        sample_by_class = {}
        all_count = 0
        waste_count = 0
        for split in cv_split:
            for cls in split:
                sample_by_class['%d' % cls] = []
                all_sample = self.coco.getImgIds(catIds=cls)
                all_count += len(all_sample)
                tqdm_gen = tqdm.tqdm(all_sample, leave=False)
                for pic in tqdm_gen:
                    if check_mask(cls, pic):
                        sample_by_class['%d' % cls].append(pic)
                    else:
                        waste_count += 1
        print(waste_count, "samples are removed.")
        return sample_by_class

    @property
    def classes(self):
        train_split = list({0, 1, 2, 3} - {self.split})
        clses = cv_split[train_split[0]] + cv_split[train_split[1]] + cv_split[train_split[2]]
        return clses

    def reset_sampler(self):
        seed = self.cfg.seed
        test_seed = self.cfg.test_seed
        # Use fixed test sampler(cfg.test_seed) for reproducibility
        self.sampler = np.random.RandomState(seed) \
            if self.train else np.random.RandomState(test_seed)

    def sample_tasks(self):
        self.tasks = []
        for i in range(len(self)):
            cls = self.sampler.choice(self.classes)
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
        if self.list is not None:
            if not self.list.exists():
                sample_list = self.generate_file_list(threshold=self.check_mask_threshold)
                with self.list.open('w') as file_obj:
                    json.dump(sample_list, file_obj)
            else:
                with self.list.open() as file_obj:
                    sample_list = json.load(file_obj)

        for i in self.classes:
            if self.list is not None:
                self.sample_by_class[i] = sample_list['%d' % i]
            else:
                self.sample_by_class[i] = self.coco.getImgIds(catIds=i)
            # Count remaining number
            self.idx_by_class[i] = len(self.sample_by_class[i])

    def __len__(self):
        return self.cfg.train_n

    def get_image(self, imgId, cache=True):
        imgId = int(imgId)
        ret_image = None

        if imgId in cache_image:
            ret_image = cache_image[imgId]
        else:
            img_meta = self.coco.loadImgs(imgId)[0]
            img = Image.open(self.img_dir / img_meta['file_name'])
            ret_image = img if img.mode == 'RGB' else img.convert("RGB")
            cache_image[imgId] = ret_image

        return ret_image

    def get_label(self, cls, imgId, cache=True, new_label=False):
        cls = int(cls)

        # Convert new label to old label
        if new_label:
            cls = cv_split[(cls - 1) // 20][(cls - 1) % 20]

        imgId = int(imgId)
        blend_name = f"{cls}_{imgId}"
        semantic_mask = None

        if blend_name in cache_label:
            semantic_mask = cache_label[blend_name]
        else:
            img_meta = self.coco.loadImgs(imgId)[0]
            annIds = self.coco.getAnnIds(imgIds=imgId)
            anns = self.coco.loadAnns(annIds)
            for ann in anns:
                catId = ann['category_id']
                if catId != cls:
                    continue
                mask = self.coco.annToMask(ann)
                if semantic_mask is not None:
                    semantic_mask[mask == 1] = 255
                else:
                    semantic_mask = np.zeros((img_meta['height'], img_meta['width']), dtype='uint8')
                    semantic_mask[mask == 1] = 255
            if cache:
                cache_label[blend_name] = semantic_mask

        return Image.fromarray(semantic_mask)

    def get_class_name(self, cls, new_label=False):
        cls = int(cls)

        # Convert new label to old label
        if new_label:
            cls = cv_split[(cls - 1) // 20][(cls - 1) % 20]



    def __getitem__(self, idx):
        height = self.cfg.height
        width = self.cfg.width
        cls, all_names = self.tasks[idx]
        sup_names = all_names[:self.shot]  # len = self.shot
        qry_names = all_names[self.shot:]

        # preprocess
        sup_rgbs, sup_masks = [], []
        for sup_name in sup_names:
            if self.train:
                scaled_factor = random.uniform(1, 1.5)
                scaled_size = (int(height * scaled_factor), int(width * scaled_factor))
                # scaled_size = (self.cfg.height, self.cfg.width)
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
                try:
                    sup_rgb, sup_mask_a = crop_obj(sup_rgb, sup_mask_a, height, width, cls, sup_name)
                except:
                    name = sup_name
                    meta = self.coco.loadImgs(name)[0]
                    print("error pic name %s, class is %d, (%s)" % (name, cls, meta['coco_url']))
                    # If meet wrong label, we provide a fake label for continuing training
                    sup_mask_a = np.zeros_like(sup_mask_a, dtype=sup_mask_a.dtype)
                    h_, w_ = sup_mask_a.shape
                    ch, cw = h_ // 2, w_ // 2
                    sup_mask_a[ch - h_ // 8:ch + h_ // 8, cw - w_ // 8:cw + w_ // 8] = 1
                    sup_rgb, sup_mask_a = crop_obj(sup_rgb, sup_mask_a, height, width, cls, sup_name)

            sup_mask_t = torch.from_numpy((sup_mask_a // 255).astype(np.float32))
            sup_mask_t = torch.stack((sup_mask_t, 1 - sup_mask_t), dim=0)
            sup_rgbs.append(sup_rgb)
            sup_masks.append(sup_mask_t)
        sup_rgb = torch.stack(sup_rgbs, dim=0)  # [shot, 3, height, width]
        sup_mask = torch.stack(sup_masks, dim=0)  # [shot, 2, height, width]

        qry_rgbs, qry_masks = [], []
        for qry_name in qry_names:
            if self.train:
                scaled_size = (height, width)
                flag = random.random()
                qry_rgb = self.normalize(self.to_tensor(
                    self.flip(flag, self.jitter(self.resize_image(scaled_size, self.get_image(qry_name))))))
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
            return (sup_rgb, sup_mask, qry_rgb), qry_mask, new_index[cls], sup_names, qry_names
        return (sup_rgb, sup_mask, qry_rgb), qry_mask, new_index[cls]


class COCOTest(COCOTrain, Dataset):
    def __init__(self, cfg, split, shot, query,
                 ret_name=False):
        Dataset.__init__(self)
        self.cfg = cfg
        self.train = False
        self.split = split
        self.shot = shot
        self.query = query
        self.base_dir = Path(self.cfg.base_dir)
        self.img_dir = self.base_dir / "val2014"
        self.annotaion_dir = self.base_dir / "annotations"
        self.coco = COCO(self.annotaion_dir / "instances_val2014.json")
        self.check_mask_threshold = 0
        self.list = self.base_dir / ("val2014_list_%d.json" % self.check_mask_threshold)
        self.ret_name = ret_name

        self.init()
        self.reset_sampler()

    @property
    def classes(self):
        clses = cv_split[self.split]
        return clses

    def __len__(self):
        return self.cfg.test_n


def load(cfg, train_mode, split, shot, query,
         bs, test_bs, num_workers, pin_memory,
         ret_name=False):
    """
    Load COCO dataset and return the batch generator.

    Parameters
    ----------
    cfg:
    train_mode: str
        train/test mode
    split: [Config]
    shot: [Config]
    query: [Config]
    bs: [Config]
    test_bs: [Config]
    num_workers: [Config]
    pin_memory: [Config]
    ret_name: bool
        Yield sample names or not.

    Returns
    -------
    dataset: Dataset instance
    data_loader: torch.utils.data.DataLoader
        For each batch:
            sup_rgb: torch.Tensor,
                Support images of the shape [bs, shot, 3, height, width] and dtype float32
            sup_mask: torch.Tensor
                PANet: Support labels of the shape [bs, shot, 2, height, width] and dtype float32
                CaNet: Support labels of the shape [bs, shot, 1, height, width] and dtype float32
            qry_rgb: torch.Tensor
                Query images of the shape [bs, query, 3, height, width] and dtype float32
            qry_mask: torch.Tensor(for train)
                Query labels of the shape [bs, query, height, width] or [bs, query, H, W] and dtype int64 (i.e. long)
            cls: list
                Index of the classes of the shape [bs]
            sup_names: list of int
                Names of support images
            qry_names: list of int
                Names of query images
    num_classes: int
        Number of classes
    """
    if train_mode == "train":
        dataset = COCOTrain(cfg, split, shot, query, ret_name=ret_name)
        data_loader = DataLoader(dataset,
                                 batch_size=bs,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=False)
    else:
        dataset = COCOTest(cfg, split, shot, query, ret_name=ret_name)
        data_loader = DataLoader(dataset,
                                 batch_size=test_bs,  # Large batch for evaluation
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=False)
    num_classes = 80
    return dataset, data_loader, num_classes


class OneExampleLoader(object):
    def __init__(self, cfg, train, sets="val"):
        self.train = train
        self.cfg = cfg
        self.base_dir = Path(cfg.base_dir)

        if sets == 'val':
            self.img_dir = self.base_dir / "val2014"
        else:
            self.img_dir = self.base_dir / "train2014"

        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        self.resize_image = lambda size, x: F.resize(x, size, interpolation=Image.BILINEAR)
        self.resize_mask = lambda size, x: F.resize(x, size, interpolation=Image.NEAREST)
        if train:
            self.anno_dir = self.base_dir / f"annotations/instances_{sets}2014.json"
            self.flip = lambda flag, x: x if flag < 0.5 else F.hflip(x)
            self.jitter = lambda x: torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)(x)
        else:
            self.anno_dir = self.base_dir / f"annotations/instances_{sets}2014.json"
        self.coco = COCO(self.anno_dir)

    def get_label(self, cls, imgId):
        img_meta = self.coco.loadImgs(imgId)[0]
        annIds = self.coco.getAnnIds(imgIds=imgId)
        anns = self.coco.loadAnns(annIds)
        semantic_masks = {}
        for ann in anns:
            catId = ann['category_id']
            if catId != cls:
                continue
            mask = self.coco.annToMask(ann)
            if catId in semantic_masks:
                semantic_masks[catId][mask == 1] = 255
            else:
                semantic_mask = np.zeros((img_meta['height'], img_meta['width']), dtype='uint8')
                semantic_mask[mask == 1] = 255
                semantic_masks[catId] = semantic_mask
        return Image.fromarray(semantic_masks[cls])

    def load(self, cls, sup_names, qry_names):
        # sup_names and qry_names should be pic ids.
        # cls should be real class index
        sup_rgbs, sup_masks = [], []
        for sup_name in sup_names:
            if self.train:
                scaled_factor = random.uniform(1, 1.5)
                scaled_size = (int(self.cfg.height * scaled_factor), int(self.cfg.width * scaled_factor))
                # scaled_size = (self.cfg.height, self.cfg.width)
                flag = random.random()
                sup_rgb = self.normalize(self.to_tensor(self.flip(flag, self.jitter(self.resize_image(
                    scaled_size, Image.open(self.img_dir / self.coco.loadImgs(sup_name)[0]['file_name']))))))
                sup_mask_a = np.array(self.flip(flag, self.resize_mask(
                    scaled_size, self.get_label(cls, sup_name))), np.uint8)
            else:
                scaled_size = (self.cfg.height, self.cfg.width)
                sup_rgb = self.normalize(self.to_tensor(self.resize_image(
                    scaled_size, Image.open(self.img_dir / self.coco.loadImgs(sup_name)[0]['file_name']))))
                sup_mask_a = np.array(self.resize_mask(
                    scaled_size, self.get_label(cls, sup_name)), np.uint8)

            if self.train:
                sup_rgb, sup_mask_a = crop_obj(sup_rgb, sup_mask_a, self.cfg.height, self.cfg.width, cls, sup_name)

            sup_mask_t = torch.from_numpy((sup_mask_a // 255).astype(np.float32))
            sup_mask_t = torch.stack((sup_mask_t, 1 - sup_mask_t), dim=0)
            sup_rgbs.append(sup_rgb)
            sup_masks.append(sup_mask_t)
        sup_rgb = torch.stack(sup_rgbs, dim=0)  # [shot, 3, height, width]
        sup_mask = torch.stack(sup_masks, dim=0)  # [shot, 2, height, width]

        qry_rgbs, qry_masks = [], []
        for qry_name in qry_names:
            if self.train:
                scaled_size = (self.cfg.height, self.cfg.width)
                flag = random.random()
                qry_rgb = self.normalize(self.to_tensor(self.flip(flag, self.jitter(self.resize_image(
                    scaled_size, Image.open(self.img_dir / self.coco.loadImgs(qry_name)[0]['file_name']))))))
                qry_mask_a = np.array(self.flip(flag, self.resize_mask(
                    scaled_size, self.get_label(cls, qry_name))), np.uint8)
            else:
                scaled_size = (self.cfg.height, self.cfg.width)
                qry_rgb = self.normalize(self.to_tensor(self.resize_image(
                    scaled_size, Image.open(self.img_dir / self.coco.loadImgs(qry_name)[0]['file_name']))))
                qry_mask_a = np.array(self.get_label(cls, qry_name), np.uint8)

            qry_mask_t = torch.from_numpy((qry_mask_a // 255).astype(np.int64))

            qry_rgbs.append(qry_rgb)
            qry_masks.append(qry_mask_t)
        qry_rgb = torch.stack(qry_rgbs, dim=0)  # [query, 3, height, width]
        qry_mask = torch.stack(qry_masks, dim=0)  # [query, height, width] / [query, H, W]

        return (sup_rgb[None], sup_mask[None], qry_rgb[None]), qry_mask[None], [cls]
