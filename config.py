import torch
import torchvision.transforms as T
class Config:
    backbone = 'fmobile'  # [resnet, fmobile]
    metric = 'arcface'  # [cosface, arcface]
    restore = False
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
    train_root = './data/CASIA-WebFace'
    test_root = "./data/lfw-align-128"
    test_list = "./data/lfw_test_pair.txt"
    checkpoints = "checkpoints"
    test_model = "checkpoints/15.pth"
    train_batch_size = 64
    test_batch_size = 64
    epochs = 20
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 5
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss'  # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = True
    num_workers = 4
config = Config()
