## Introduction
A PyTorch implementation of [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

## Installation

```
conda create -n vgg python=3.8
conda activate vgg
pip install -r requirements.txt
```

## Architectures
- VGG-A
- VGG-A-LRN
- VGG-B
- VGG-C
- VGG-D
- VGG-E

## Dataset
- [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)

## Training
```
python train.py --model vgg_A --dataset cifar10
```

## Eval
```
python eval.py --weights WEIGHTS_FILEPATH --dataset cifar10
```

## Inference
```
python infer.py --weights WEIGHTS_FILEPATH --image IMAGE_FILEPATH
```
