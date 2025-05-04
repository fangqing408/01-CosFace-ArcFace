# 01-CosFace-ArcFace

## 1 >> 依赖环境

本项目主要用到了 Pytorch 库，硬件配置和项目环境如下。

- 硬件配置

```
RTX 3070Ti
cuda126 + cuDNN
```

- 项目环境

```
numpy                     1.26.3
pillow                    11.0.0
python                    3.9.21
torch                     2.6.0+cu126
torchvision               0.21.0+cu126
tqdm         
```

注：tqdm 也不是必须的，为训练过程添加一个进度条，方便查看训练的进度。

## 2 >> 项目参数

定义了项目训练过程中的参数，方便配置的修改。本文主要研究的是 cosface、arcface 角边距损失对人脸识别的影响，故 metric 参数选用了 cosface 和 arcface 两种。

```python
import torch
import torchvision.transforms as T
class Config:
    backbone = 'fmobile'  # [resnet, fmobile]
    metric = 'arcface'  # [cosface, arcface]
    ...
```

接着定义了 embedding_size，这个参数定义了多长的特征向量表示一张人脸，接下来就是对于训练图片和测试图片的预处理，要是写成一个加载图片的类的话，可以在 `__getitem__` 里面预处理图片。

对于训练图片，先进行灰度转换，然后随机翻转，更改大小，随即剪切，转化为张量类型，标准化。

对于测试图片，先进行灰度转换，然后更改大小，转化为张量类型，标准化到 [-1, 1] 之间

```python
    ...
    embedding_size = 512
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    ...
```

下面的是一些常见的文件位置，训练测试批次大小，训练轮数，优化器，学习率和损失函数类型等参数。

比较重要的是 device 参数，一般都是文中这样写，没有 GPU 的时候会自动选择 CPU 进行训练，pin_memory 能加快内存到 GPU 之间的传输，num_workers 不为 0 是启用多进程加载。

```python
    ...
    train_root = './data/CASIA-WebFace'
    test_root = "./data/lfw-align-128"
    test_list = "./data/lfw_test_pair.txt"
    checkpoints = "checkpoints"
    test_model = "checkpoints/3.pth"
    train_batch_size = 64
    test_batch_size = 64
    epochs = 10
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss'  # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = True
    num_workers = 4
config = Config()
```


## 3 >> 构建步骤

- 001 >> [数据准备](https://github.com/fangqing408/01-CosFace-ArcFace/blob/master/recognition/001.md)
- 002 >> [网络构建](https://github.com/fangqing408/01-CosFace-ArcFace/blob/master/recognition/002.md)
- 003 >> [损失函数](https://github.com/fangqing408/01-CosFace-ArcFace/blob/master/recognition/003.md)
- 004 >> [角边距函数](https://github.com/fangqing408/01-CosFace-ArcFace/blob/master/recognition/004.md)
- 005 >> [模型训练](https://github.com/fangqing408/01-CosFace-ArcFace/blob/master/recognition/005.md)
- 006 >> [模型测试](https://github.com/fangqing408/01-CosFace-ArcFace/blob/master/recognition/006.md)
