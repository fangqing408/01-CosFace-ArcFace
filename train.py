import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import FaceMobileNet
from model.metric import CosFace, ArcFace
from model.loss import FocalLoss
from dataset import load_data
from config import config as conf
if __name__ == '__main__':
    dataloader, class_num = load_data(conf, training=True)
    device = conf.device
    embedding_size = conf.embedding_size
    net = FaceMobileNet(embedding_size).to(device)
    metric = CosFace(embedding_size, class_num).to(device)
    criterion = FocalLoss(gamma=2)
    optimizer = optim.SGD(
        list(net.parameters()) + list(metric.parameters()),
        lr=conf.lr,
        weight_decay=conf.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=conf.lr_step, gamma=0.1)
    if conf.restore:
        net.load_state_dict(torch.load(conf.restore_model, map_location=device))
    net.train()
    for e in range(conf.epochs):
        total_loss = 0
        for data, labels in tqdm(dataloader, desc=f"Epoch {e+1}/{conf.epochs}", ascii=True):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = net(data)
            thetas = metric(embeddings, labels)
            loss = criterion(thetas, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {e+1}/{conf.epochs}, Loss: {avg_loss:.4f}")
        torch.save(net.state_dict(), f"./checkpoints/{e}.pth")
        scheduler.step()
