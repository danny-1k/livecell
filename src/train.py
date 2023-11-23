import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from models import SegNet, Unet
from data import LiveCellDataset


class Trainer:
    def __init__(self, num_epochs:int, net:nn.Module, optimiser:optim.Optimizer, lossfn:nn.Module, use_tensorboard:bool) -> None:
        self.num_epochs = num_epochs
        self.net = net
        self.optimiser = optimiser
        self.lossfn = lossfn
        self.use_tensorboard = use_tensorboard

        self.history = {}
        pass

    def train_one_epoch(self):
        pass

    def test_one_epoch(self):
        pass

    def run(self):
        pass


patch_size = (128, 128)
batch_size = 8

net = SegNet()

train = LiveCellDataset(split="train", patch_size=patch_size)
test = LiveCellDataset(split="test", patch_size=patch_size)

train = DataLoader(train, batch_size=batch_size, shuffle=True)
test = DataLoader(test, batch_size=batch_size, shuffle=True)