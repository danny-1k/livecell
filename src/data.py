import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from einops import rearrange

from pycocotools.coco import COCO

from PIL import Image


class LiveCellDataset(Dataset):
    def __init__(self, split:str, patch_size:Optional[tuple]=None) -> None:
        if split not in ["test", "train", "val"]:
            raise ValueError("Invalid value for split")
        self.split = split
        self.coco = COCO(f"../data/ann/livecell_coco_{split}.json")
        self.img_ids = self.coco.getImgIds()
        self.patch_size = patch_size

        self.transform = transforms.Compose([
            lambda x: torch.from_numpy(x), #transforms.ToTensor(),
            lambda x: (x-127.5)/127.5 # -> x E [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.coco.loadImgs(self.img_ids[index])[0]
        ann_ids = self.coco.getAnnIds(imgIds=img["id"])
        anns = self.coco.loadAnns(ann_ids)

        mask = np.sum([self.coco.annToMask(ann) for ann in anns], axis=0) # sum up mask to create a single mask of indices
        mask[mask>=2] = 2

        # A value of 0 represents background, 1 represents part of cell and >=2 represents intersection between 2 or more cells

        img = np.asarray(Image.open("../data/imgs/images/{}/{}".format(
                f"livecell_{'test' if self.split == 'test' else 'train_val'}_images",
                img["file_name"]
            )
        ).convert("L"))

        # splitting the image (and mask) into patches might be benefitial

        if self.patch_size:
            # First check if the patch size can divide the image perfectly. If not, we pad the image and mask.

            if img.shape[0] % self.patch_size[0] != 0:
                r = math.ceil(img.shape[0]/self.patch_size[0])
                p = int(r * self.patch_size[0]) - img.shape[0]
                
                # if padding is odd, then add it to the bottom of the image
                # if the padding is even then add half on the top and half on the bottom

                if p%2 == 0:
                    img2 = np.zeros((img.shape[0]+p, img.shape[1]))
                    mask2 = np.zeros((img.shape[0]+p, img.shape[1]))

                    img2[p//2: p//2 + img.shape[0], :] = img
                    mask2[p//2: p//2 + img.shape[0], :] = mask

                    img = img2
                    mask = mask2
                    
                    del img2
                    del mask2

                else:
                    img2 = np.zeros((img.shape[0]+p, img.shape[1]))
                    mask2 = np.zeros((img.shape[0]+p, img.shape[1]))

                    img2[:img.shape[0], :] = img
                    mask2[:img.shape[0], :] = mask

                    img = img2
                    mask = mask2

                    del img2
                    del mask2

            if img.shape[1] // self.patch_size[1] != 0:
                r = math.ceil(img.shape[1]/self.patch_size[1])
                p = int(r * self.patch_size[1]) - img.shape[1]

                # if padding is odd, then add it to the right of the image
                # if the padding is even then add half on the left and half to the right

                if p%2 == 0:
                    img2 = np.zeros((img.shape[0], img.shape[1]+p))
                    mask2 = np.zeros((img.shape[0], img.shape[1]+p))

                    img2[:, p//2: p//2 + img.shape[1]] = img
                    mask2[:, p//2: p//2 + img.shape[1]] = mask

                    img = img2
                    mask = mask2

                    del img2
                    del mask2

                else:
                    img2 = np.zeros((img.shape[0], img.shape[1]+p))
                    mask2 = np.zeros((img.shape[0], img.shape[1]+p))

                    img2[:, :img.shape[1]] = img
                    mask2[:, :img.shape[1]] = mask


                    img = img2
                    mask = mask2

                    del img2
                    del mask2

            img = rearrange(img, "(n1 p1) (n2 p2) -> (n1 n2) p1 p2", p1=self.patch_size[0], p2=self.patch_size[1])
            mask = rearrange(mask, "(n1 p1) (n2 p2) -> (n1 n2) p1 p2", p1=self.patch_size[0], p2=self.patch_size[1])

            img = self.transform(np.asarray(img)).float()
            img = img.unsqueeze(1)
            mask = torch.from_numpy(mask).long()

        return img, mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    data = LiveCellDataset(
        split="train",
        patch_size=(128, 128)
    )

    img, mask = data[100]

    n = img.shape[0]

    for i in range(n):

        plt.imshow(img[i])
        plt.savefig(f"../img_{i}.png")
        plt.close("all")

        plt.imshow(mask[i])
        plt.savefig(f"../mask_{i}.png")
        plt.close("all")
