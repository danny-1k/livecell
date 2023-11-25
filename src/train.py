import os
from typing import List
from datetime import datetime

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import SegNet, Unet
from data import LiveCellDataset

from averager import Averager

from metrics import Metric

def setup():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"../runs/run_{timestamp}"
    save_dir = f"../checkpoints/{timestamp}"

    if not os.path.exists("../runs"):
        os.makedirs("../runs")

    if not os.path.exists("../checkpoints"):
        os.makedirs("../checkpoints")

    os.makedirs(run_name)
    os.makedirs(save_dir)

    return run_name, save_dir


class Trainer:
    def __init__(self,
                num_epochs:int,
                net:nn.Module, 
                optimiser, 
                lossfn:nn.Module, 
                use_tensorboard:bool, 
                device:str,
                metrics:list[Metric]) -> None:
        
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.net = net
        self.optimiser = optimiser(self.net)
        self.lossfn = lossfn
        self.use_tensorboard = use_tensorboard
        self.device = device

        self.history = {}

    def train_one_epoch(self, data):
        self.net.train()

        loss_averager = Averager()
        metrics_averager = [metric.new() for metric in self.metrics]

        for x, y in data:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimiser.zero_grad()

            # x of shape (N, n, h, w)
            # y of shape (N, n, h, w) -> (N, h, w)

            x = x.view(-1, x.shape[-2], x.shape[-1])
            y = y.view(-1, y.shape[-2], y.shape[-1])

            p = net(x)

            loss = self.lossfn(p.permute(0, 2, 3, 1).view(-1, p.shape[1]), y.view(-1))

            print(loss.item())

            loss.backward()

            self.optimiser.step()

            loss_averager += loss.item()
            
            for m in metrics_averager:
                m += m.eval(p, y)

        return loss_averager, metrics_averager

    def test_one_epoch(self):
        self.net.eval()
        pass

    def run(self, train, test):
        self.net.to(self.device)
        train_loss,_averager, test_metrics_averager = self.train_one_epoch(train)

    def save(self):
        # save model checkpoint
        # save optimiser checkpoint

        pass

lr = 1e-4

patch_size = (64, 64)
batch_size = 1
epochs = 100

net = SegNet(
    in_channels=1,
    out_channels=16,
    channel_expansion=2,
    num_blocks=3,
    num_classes=3,
)

train = LiveCellDataset(split="train", patch_size=patch_size)
test = LiveCellDataset(split="test", patch_size=patch_size)

train = DataLoader(train, batch_size=batch_size, shuffle=True)
test = DataLoader(test, batch_size=batch_size, shuffle=True)

lossfn = nn.CrossEntropyLoss()

optimiser =  lambda net: optim.Adam(net.parameters(), lr=lr)


trainer = Trainer(num_epochs=1, net=net, optimiser=optimiser, lossfn=lossfn, use_tensorboard=False, device="cpu", metrics=[])

trainer.run(train, test)

# for epoch in range(epochs):
#     net.train()
#     for x, y in train:
#         optimiser.zero_grad()

#         # x of shape (N, n, h, w)
#         # y of shape (N, n, h, w) -> (N, h, w)

#         x = x.view(-1, x.shape[-2], x.shape[-1])
#         y = y.view(-1, y.shape[-2], y.shape[-1])

#         p = net(x)
        
#         # p of shape (N, C, h, w) -> (N, C)
#         # y of shape (N, h, w) -> (N)

#         loss = lossfn(p.permute(0, 2, 3, 1).view(-1, p.shape[1]), y.view(-1))

#         loss.backward()

#         optimiser.step()

#     net.eval()

#     for x,y in test:
#         optimiser.zero_grad()

#         x = x.view(-1, x.shape[-2], x.shape[-1])
#         y = y.view(-1, y.shape[-2], y.shape[-1])

#         p = net(x)

#         loss = lossfn(p.permute(0, 2, 3, 1).view(-1, p.shape[1]), y.view(-1))

