from typing import Optional, List, Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, activation:Optional[nn.Module]=nn.ReLU):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding),
            activation(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return x


class Unet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass

class SegNetEncoder(nn.Module):
    def __init__(self, layers:List[torch.Tensor]):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.downsample = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        indices = []

        for layer in self.layers:
            x = layer(x)
            x, x_indices = self.downsample(x)
            indices.append(x_indices)

        return x, indices


class SegNetDecoder(nn.Module):
    def __init__(self, layers:List[torch.Tensor]):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.upsample = nn.MaxUnpool2d(2, 2)

    def forward(self, x:torch.Tensor, indices) -> List[torch.Tensor]:
        # the indices are being used in reverse order in the decoder, so we have to reverse the list of indices
        indices = indices[::-1]
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.upsample(x, indices[idx])
        
        return x


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels, channel_expansion, num_classes, num_blocks):
        super().__init__()

        encoder_layers = []
        decoder_layers = []

        prev_channels = None

        for n in range(num_blocks):
            encoder_layers.append(
                ConvBlock(
                    in_channels=prev_channels or in_channels, 
                    out_channels=out_channels if n==0 else out_channels*n*channel_expansion,
                    kernel_size=3,
                    stride=1, 
                    padding=1
                )
            )

            prev_channels = out_channels if n==0 else out_channels*n*channel_expansion

        prev_channels = None

        for n in list(range(num_blocks))[::-1]:
            decoder_layers.append(
                ConvBlock(
                    in_channels=prev_channels or out_channels*n*channel_expansion, 
                    out_channels=out_channels if n==0 else out_channels*n*channel_expansion,
                    kernel_size=3,
                    stride=1, 
                    padding=1
                )
            )

            prev_channels = out_channels if n==0 else out_channels*n*channel_expansion

        self.encoder = SegNetEncoder(layers=encoder_layers)
        self.decoder = SegNetDecoder(layers=decoder_layers)

        self.project = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0) # final 1x1 conv to project to desired num of classes

    def forward(self, x:torch.Tensor):
        x, indices = self.encoder(x)
        x = self.decoder(x, indices)
        x = self.project(x)

        return x


if __name__ == "__main__":

    x = torch.zeros((1, 1, 128, 128))

    segnet = SegNet(in_channels=1, out_channels=16, channel_expansion=2, num_classes=3, num_blocks=3)

    p = segnet(x)

    print(p.shape)