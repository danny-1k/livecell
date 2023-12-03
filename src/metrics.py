import torch
import torch.nn.functional as F


class Metric:
    def __init__(self) -> None:
        self.name = None
        self.value = 0

    def __iadd__(self, value:float) -> None:
        self.value = .9*value + .1*self.value
    
    @torch.no_grad()
    def eval(p:torch.Tensor, y:torch.Tensor) -> float:
        raise NotImplementedError
    
    @classmethod
    def new(self):
        return self()


class DICE(Metric):
    def __init__(self, classes=3) -> None:
        super().__init__()

        self.name = "DICE"
        self.classes = classes

    @torch.no_grad()
    def eval(self, p:torch.Tensor, y:torch.Tensor) -> float:
        # p of shape (N, H, W)
        # y of shape (N, H, W)

        p = F.one_hot(p, self.classes) # (N, H, W, C)
        y = F.one_hot(y, self.classes) # (N, H, W, C)

        p = p.permute(0, 3, 1, 2) # (N, C, H, W)
        y = y.permute(0, 3, 1, 2) # (N, C, H, W)

        intersection = (p & y).sum(dim=(2, 3)) # (N, C)
        union = p.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) # (N, C)

        dice = (2 * intersection + 1e-8) / (union + 1e-8)

        dice = dice.mean().item()

        return dice


class IOU(Metric):
    def __init__(self, classes=3) -> None:
        super().__init__()
        
        self.name = "IOU"
        self.classes = classes

    @torch.no_grad()
    def eval(self, p:torch.Tensor, y:torch.Tensor) -> float:
        # p of shape (N, H, W)
        # y of shape (N, H, W)

        p = F.one_hot(p, self.classes) # (N, H, W, C)
        y = F.one_hot(y, self.classes) # (N, H, W, C)

        p = p.permute(0, 3, 1, 2) # (N, C, H, W)
        y = y.permute(0, 3, 1, 2) # (N, C, H, W)

        intersection = (p & y).sum(dim=(2, 3)) # (N, C)
        union = (p | y).sum(dim=(2, 3)) # (N, C)

        iou = intersection / (union + 1e-8)

        iou = iou.mean().item()

        return iou