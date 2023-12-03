import os
from typing import List, Optional
from datetime import datetime

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter


from averager import Averager

from metrics import Metric

from tqdm import tqdm

def setup(dir):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{dir}/runs/run_{timestamp}"
    save_dir = f"{dir}/checkpoints/{timestamp}"

    if not os.path.exists(f"{dir}/runs"):
        os.makedirs(f"{dir}/runs")

    if not os.path.exists(f"{dir}/checkpoints"):
        os.makedirs(f"{dir}/checkpoints")

    os.makedirs(run_name)
    os.makedirs(save_dir)

    return run_name, save_dir


class Trainer:
    def __init__(self,
                num_epochs:int,
                net:nn.Module, 
                optimiser, 
                lossfn:nn.Module, 
                # use_tensorboard:bool, 
                device:str,
                dir:str,
                metrics:list[Metric],
                checkpoint:Optional[str]=None) -> None:
        
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.net = net
        self.optimiser = optimiser(self.net)
        self.lossfn = lossfn
        # self.use_tensorboard = use_tensorboard
        self.device = device

        self.run_name, self.save_dir = setup(dir)

        self.history = {}

        if checkpoint:
            try:
                checkpoint = torch.load(checkpoint, map_location=next(self.net.parameters()).device)
                self.net.load_state_dict(checkpoint["checkpoint"])
                self.optimiser.load_state_dict(checkpoint["optimiser"])
                
                print("Successfully Loaded Checkpoint")
                print(f"\t\t Epoch -> {checkpoint['epoch']}")
                print(f"\t\t Train Loss -> {checkpoint['train_loss']}")
                print(f"\t\t Test Loss -> {checkpoint['test_loss']}")

            except:
                print("Could load load checkpoint...")

    def train_one_epoch(self, data):
        self.net.train()

        loss_averager = Averager()
        metrics_averager = [metric.new() for metric in self.metrics]

        for x, y in tqdm(data):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimiser.zero_grad()

            # x of shape (N, n, h, w)
            # y of shape (N, n, h, w) -> (N, h, w)

            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            y = y.view(-1, y.shape[-2], y.shape[-1])

            p = net(x)

            loss = self.lossfn(p.permute(0, 2, 3, 1).contiguous().view(-1, p.shape[1]), y.view(-1))

            loss.backward()

            self.optimiser.step()

            loss_averager += loss.item()
            
            for m in metrics_averager:
                m += m.eval(p.argmax(dim=1), y)

        return loss_averager, metrics_averager

    def test_one_epoch(self, data):
        self.net.eval()

        loss_averager = Averager()
        metrics_averager = [metric.new() for metric in self.metrics]

        for x, y in tqdm(data):
            x = x.to(self.device)
            y = y.to(self.device)

            # x of shape (N, n, h, w)
            # y of shape (N, n, h, w) -> (N, h, w)

            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            y = y.view(-1, y.shape[-2], y.shape[-1])

            p = net(x)

            loss = self.lossfn(p.permute(0, 2, 3, 1).contiguous().view(-1, p.shape[1]), y.view(-1))

            loss_averager += loss.item()
            
            for m in metrics_averager:
                m += m.eval(p.argmax(dim=1), y)

        return loss_averager, metrics_averager
    
    def run(self, train, test):
        self.net.to(self.device)
        self.optimiser.to(self.device)

        lowest_loss = None

        for epoch in range(self.num_epochs):
            train_loss_averager, train_metrics_averager = self.train_one_epoch(train)
            test_loss_averager, test_metrics_averager = self.test_one_epoch(test)

            if not lowest_loss or test_loss_averager.value < lowest_loss:
                lowest_loss = test_loss_averager.value

                checkpoint = {
                    "epoch": epoch+1,
                    "checkpoint": self.net.state_dict(),
                    "optimiser": self.optimiser.state_dict(),
                    "train_loss": train_loss_averager.value,
                    "test_loss": test_loss_averager.value,
                }

                for metric in train_metrics_averager:
                    checkpoint[f"train_{metric.name}"] = metric.value

                for metric in test_metrics_averager:
                    checkpoint[f"test_{metric.name}"] = metric.value

                torch.save(checkpoint, f"{self.save_dir}/checkpoint.pt")

            print(f"EPOCH: {epoch} | TRAIN_LOSS: {train_loss_averager.value:.3f} {' '.join([f'| TRAIN_{metric.name}: {metric.value}' for metric in train_metrics_averager])} TEST_LOSS: {test_loss_averager.value:.3f} {' '.join([f'| TEST_{metric.name}: {metric.value}' for metric in test_metrics_averager])}")


if __name__ == "__main__":

    from argparse import ArgumentParser

    from torch.utils.data import DataLoader
    from metrics import IOU, DICE

    from models import SegNet, Unet
    from data import LiveCellDataset

    parser = ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dir", type=str, default="..")

    parser.add_argument("--out_channels", type=int, default=16)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=3) # NOTE: Images are split into patches and total batch size is N * number of patches
    parser.add_argument("--model", type=str, default="segnet")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    
    if args.model == "segnet":
        net = SegNet(
            in_channels=1,
            out_channels=args.out_channels,
            channel_expansion=args.expansion,
            num_blocks=args.blocks,
            num_classes=3,
        )

    else:
        net = Unet()

    optimiser = lambda net: optim.Adam(net.parameters(), lr=args.lr)
    
    trainer = Trainer(
        num_epochs=args.epochs,
        net=net,
        optimiser=optimiser,
        lossfn=nn.CrossEntropyLoss(),
        device=args.device,
        dir=args.dir,
        metrics=[IOU, DICE],
        checkpoint=args.resume
    )

    train = LiveCellDataset(split="train", patch_size=(args.patch_size, args.patch_size))
    test = LiveCellDataset(split="test", patch_size=(args.patch_size, args.patch_size))

    train = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test = DataLoader(test, batch_size=args.batch_size, shuffle=True)

    trainer.run(train, test)