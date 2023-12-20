import math
import torch
from torchvision.transforms import transforms
from einops import rearrange
from models import SegNet
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np


def to_patches(img, patch_size):
    padding = [
        (), # top, bottom
        (), # left, right
    ]

    if img.shape[0] % patch_size[0] != 0:
        r = math.ceil(img.shape[0]/patch_size[0])
        p = int(r * patch_size[0]) - img.shape[0]
        
        # if padding is odd, then add it to the bottom of the image
        # if the padding is even then add half on the top and half on the bottom

        if p%2 == 0:
            img2 = np.zeros((img.shape[0]+p, img.shape[1]))

            img2[p//2: p//2 + img.shape[0], :] = img

            img = img2
            
            del img2

            padding[0] = (p//2, p//2)

        else:
            img2 = np.zeros((img.shape[0]+p, img.shape[1]))

            img2[:img.shape[0], :] = img

            img = img2

            del img2

            padding[0] = (0, p)

    if img.shape[1] // patch_size[1] != 0:
        r = math.ceil(img.shape[1]/patch_size[1])
        p = int(r * patch_size[1]) - img.shape[1]

        # if padding is odd, then add it to the right of the image
        # if the padding is even then add half on the left and half to the right

        if p%2 == 0:
            img2 = np.zeros((img.shape[0], img.shape[1]+p))

            img2[:, p//2: p//2 + img.shape[1]] = img

            img = img2

            del img2

            padding[1] = (p//2, p//2)

        else:
            img2 = np.zeros((img.shape[0], img.shape[1]+p))

            img2[:, :img.shape[1]] = img


            img = img2

            del img2

            padding[1] = (0, p)

        
    return img, padding


checkpoint_f = "../checkpoints/checkpoint.pt"
test_img = "../data/imgs/images/livecell_test_images/A172_Phase_C7_1_00d00h00m_1.tif"

checkpoint = torch.load(checkpoint_f, map_location=torch.device("cpu"))

print(f"Loading checkpoint from Epoch {checkpoint['epoch']} with train IOU {checkpoint['train_IOU'] :.3f} test IOU {checkpoint['test_IOU']:.3f} train DICE {checkpoint['train_DICE']:.3f} test DICE {checkpoint['test_DICE']:.3f}")

out_channels = 16
expansion = 2
blocks = 3
patch_size = 128


net = SegNet(
    in_channels=1,
    out_channels=out_channels,
    channel_expansion=expansion,
    num_blocks=blocks,
    num_classes=3
)
net.eval()
net.load_state_dict(checkpoint["checkpoint"])

# img = rearrange(img, "(n1 p1) (n2 p2) -> (n1 n2) p1 p2", p1=patch_size[0], p2=patch_size[1])
#             mask = rearrange(mask, "(n1 p1) (n2 p2) -> (n1 n2) p1 p2", p1=patch_size[0], p2=patch_size[1])

img = np.asarray(Image.open(test_img).convert("L"))
img_size = img.shape
img, padding = to_patches(img, patch_size=(patch_size, patch_size))
img = rearrange(img, "(n1 p1) (n2 p2) -> (n1 n2) p1 p2", p1=patch_size, p2=patch_size)

transform = transforms.Compose([
            lambda x: torch.from_numpy(x), #transforms.ToTensor(),
            lambda x: (x-127.5)/127.5 # -> x E [-1, 1]
        ])


img = transform(img)
img = img.unsqueeze(1).float()

p = net(img)

p = p.argmax(1)

# p of shape (n, p1, p2) -> (p1 * (h/p1), p2 * (w/p2))

print(p.shape)

p = p.detach().numpy()

# break paddings off

vertical_padding, horizontal_padding = padding

print(img_size, "img size")

new_size = (img_size[0] + sum(vertical_padding), img_size[1] + sum(horizontal_padding))

print(new_size)
p = p.reshape(((new_size[0]//128)*128), -1)

# img = rearrange(p, "(n1 n2) p1 p2 -> (n1 p1) (n2 p2)", p1=128, p2=128, n1=15, n2=2)

print(p.shape)
plt.imshow(p)
plt.show()

# print(img.shape)