# DTPC
This is the code for "Semi-Supervised Segmentation of Medical Image via Dynamic Thresholding and Prototype Contrastive Learning"

## Getting Started

### Dataset download
- ISIC 2018: [image and mask](https://challenge.isic-archive.com/data/#2018)
- Kvasir-SEG: [image and mask](https://datasets.simula.no/kvasir-seg/)
- ACDC: [image and mask](https://drive.google.com/file/d/1LOust-JfKTDsFnvaidFOcAVhkZrgW1Ac/view?usp=sharing)

you can download datasets and put into your dataset path, like:
```
├── [Your Dataset Path]
    ├── ISIC 2018
    ├── Kvasir-SEG
    └── ACDC
```
对于ISIC 2018与Kvasir-SEG，在训练之前，还需要对数据进行resize与train/val划分的预处理。你可以运行以下预处理函数：
```
cd code
python isic2018_processing.py and kvasirSEG_processing.py
```
半监督学习的label/unlabel存储在``splits``文件中：
```
├── splits
    ├── ISIC2018
        ├──1_4
          ├──labeled.txt
          ├──unlabeled.txt
        ├──1_8
          ├──labeled.txt
          ├──unlabeled.txt
        ├──1_16
          ├──labeled.txt
          ├──unlabeled.txt
        ├──train.txt
        ├──val.txt
    ├── Kvasir-SEG
        ├──1_4
          ├──labeled.txt
          ├──unlabeled.txt
        ├──1_8
          ├──labeled.txt
          ├──unlabeled.txt
        ├──1_16
          ├──labeled.txt
          ├──unlabeled.txt
        ├──train.txt
        ├──val.txt
    └── ACDC
        ├──3
          ├──labeled.txt
          ├──unlabeled.txt
        ├──7
          ├──labeled.txt
          ├──unlabeled.txt
        ├──test.txt
        ├──val.txt
        ├──valtest.txt
        ├──train_slices.txt
```
### Train the model
为了超参数等配置方便，我们将训练参数统一使用``yaml``文件进行配置，统一存放在``configs``文件中。

If you want train model on ISIC 2018 or Kvasir-SEG, you can do:
```
cd code
python train_2d_DTPC.py
```

If you want train model on ACDC, you can do:
```
cd code
python train_2d_DTPC_ACDC.py
```

## Acknowledgement

This code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [UniMatch](https://github.com/LiheYoung/UniMatch).
We thank Xiangde Luo and Lihe Yang for their elegant and efficient code base.
