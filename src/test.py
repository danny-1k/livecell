import os
import math
import torch
from torchvision.transforms import transforms
from einops import rearrange
from models import SegNet
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

import cv2 as cv


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


    num_vertical = img.shape[0] // patch_size[0]
    num_horizontal = img.shape[1] // patch_size[1]

    out = np.zeros((num_vertical*num_horizontal, patch_size[0], patch_size[1]))

    for i in range(num_vertical):
        for j in range(num_horizontal):
            out[i*num_horizontal + j] = img[i*patch_size[0]: i*patch_size[0] + patch_size[0], j*patch_size[1]: j*patch_size[1] + patch_size[1]]

    return out, padding


def stitch_patches(patch_size, img_size, padding, patches):
    vertical_padding, horizontal_padding = padding

    new_img_size = (img_size[0] + sum(vertical_padding), img_size[1] + sum(horizontal_padding))
    
    num_vertical = new_img_size[0] // patch_size[0]
    num_horizontal = new_img_size[1] // patch_size[1]
    
    out_img = np.zeros(new_img_size)

    for i in range(num_vertical):
        for j in range(num_horizontal):
            out_img[i*patch_size[0]: i*patch_size[0] + patch_size[0], j*patch_size[1]: j*patch_size[1] + patch_size[1]] = patches[i*num_horizontal + j]

    # strip padding
    out_img = out_img[vertical_padding[0]:-vertical_padding[1], horizontal_padding[0]:-horizontal_padding[1]]

    return out_img


#https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay
def overlay_mask(img, mask, color, alpha):
    img = np.expand_dims(img, 0).repeat(3, axis=0) # make it 3 channels

    color = np.array(color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    image_combined = cv.addWeighted(img, 1-alpha, image_overlay, alpha, 0)

    return image_combined


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    
    parser.add_argument("--f", required=True)
    parser.add_argument("--checkpoint", default="../checkpoints/checkpoint.pt")
    
    parser.add_argument("--expansion", default=2, type=int)
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--out_channels", default=16, type=int)
    parser.add_argument("--patch_size", default=128, type=int)
    parser.add_argument("--pred_dir", default="../pred", type=str)
    
    args = parser.parse_args()

    test_img_f = args.f
    checkpoint_f = args.checkpoint
    expansion = args.expansion
    num_blocks = args.num_blocks
    out_channels = args.out_channels
    patch_size = (args.patch_size, args.patch_size)
    pred_dir = args.pred_dir

    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    checkpoint = torch.load(checkpoint_f, map_location=torch.device("cpu"))

    print(f"Loading checkpoint from Epoch {checkpoint['epoch']} with train IOU {checkpoint['train_IOU'] :.3f} test IOU {checkpoint['test_IOU']:.3f} train DICE {checkpoint['train_DICE']:.3f} test DICE {checkpoint['test_DICE']:.3f}")

    net = SegNet(
        in_channels=1,
        out_channels=out_channels,
        channel_expansion=expansion,
        num_blocks=num_blocks,
        num_classes=3
    )
    
    net.eval()
    net.load_state_dict(checkpoint["checkpoint"])

    img = np.asarray(Image.open(test_img_f).convert("L"))

    pre_img = img.copy()
    img_size = img.shape
    img, padding = to_patches(img, patch_size=patch_size)

    transform = transforms.Compose([
        lambda x: torch.from_numpy(x), #transforms.ToTensor(),
        lambda x: (x-127.5)/127.5 # -> x E [-1, 1]
    ])

    img = transform(img)
    img = img.unsqueeze(1).float()

    patches = net(img).argmax(1).detach().numpy()

    mask = stitch_patches(patch_size=patch_size, img_size=img_size, padding=padding, patches=patches)
    mask = (mask != 0)

    overlayed_img = overlay_mask(pre_img, mask, color=(255, 0, 0), alpha=.3)

    prediction_filename = test_img_f.replace("\\", "/").split("/")[-1].split(".")[0]
    prediction_filename = f"{prediction_filename}.png"
    
    cv.imwrite(os.path.join(pred_dir, prediction_filename), overlayed_img.transpose(1, 2, 0))

    plt.imshow(overlayed_img.transpose(1, 2, 0))
    plt.show()