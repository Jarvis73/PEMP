# Data & Pre-trained Models

Put the datasets and the pre-trained models here.

### 1. Prepare datasets

*   [x] PASCAL-5i
*   [x] COCO-20i

#### 1.1 PASCAL-5i

*   Download [Training/Validation data (2G)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), and extract `VOCtrainval_11-May-2012.tar` to `./data/`  
*   Download [CaNet annotations (20.7M)](https://github.com/icoz69/CaNet/raw/master/Binary_map_aug.zip), and extract `Binary_map_aug.zip` to `./data/VOCdevkit/VOC2012/` 

#### 1.2 COCO-20i

*   Download [2014 Training images (13GB)](http://images.cocodataset.org/zips/train2014.zip), [2014 Val images (6GB)](http://images.cocodataset.org/zips/val2014.zip), [2014 Train/Val annotations (241M)](http://images.cocodataset.org/annotations/annotations_trainval2014.zip), and extract them to `./data/COCO` 

### 2. Pre-trained weights

*   [x] [VGG-16](https://download.pytorch.org/models/vgg16-397923af.pth) 
*   [x] [Resnet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth)
*   [x] [Resnet-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
*   [x] [Resnet50_v2](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) (for PFENet. See the [PFENet repository](https://github.com/Jia-Research-Lab/PFENet) for details.)

*   Download pre-trained models and put them into `./data` .

Final directory structure (only display used directories and files):

```
./data
├── COCO
│   ├── annotations
│   ├── train2014
│   └── val2014
├── VOCdevkit
│   └── VOC2012
│       ├── Binary_map_aug
│       └── JPEGImages
├── README.md
├── resnet101-5d3b4d8f.pth
├── resnet50-19c8e357.pth
├── resnet50_v2.pth
└── vgg16-397923af.pth
```


