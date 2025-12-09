import torch
from PIL import Image
import argparse
import os
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision.transforms import transforms
from pytorch_fid import fid_score
import random

import torch.nn as nn

parser = argparse.ArgumentParser(description="Generate images with stable diffusion")
parser.add_argument("--dir_path", type=str, default="_exp/input/FFHQ")
parser.add_argument(
    "--dest_folder",
    type=str,
    default="_exp/output/dwgf/in",
)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--seed", type=int, default=100)
args = parser.parse_args()

torch.cuda.set_device(0)

dest_folder = args.dest_folder

it = 0
for item in os.listdir(args.dir_path):
    list_imgs = []
    if item != "FID":
        if os.listdir(os.path.join(args.dir_path, item)) == []:
            continue
        for img_id in os.listdir(os.path.join(args.dir_path, item)):
            # print(item)
            list_imgs.append(img_id)
            # Remove element that has (img_id.split('_')[2]).split('.')[0] != 'grid' in its name from list_imgs

        print(item)
        # list_imgs = [x for x in list_imgs if (x.split('_')[2]).split('.')[0] != 'grid']
        list_imgs = [x for x in list_imgs if len(x.split("_")) == 1]

        # print(item)
        print(list_imgs)
        # idx_img = np.random.choice(len(list_imgs), size=3)
        idx_img = np.random.choice(len(list_imgs))
        img_true = Image.open(
            os.path.join(args.dir_path, item, list_imgs[idx_img])
        ).convert("RGB")

        img_torch = transforms.ToTensor()(img_true)
        img_torch = transforms.Resize((512, 512))(img_torch)
        # Create folder if does not exist
        # dest_folder = args.dest_folder + f'/{item}'
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

    # save_image(img_torch, f'{dest_folder}/x_{0}.png')
    save_image(img_torch, f"{dest_folder}/{item}.png")

    it = it + 1


# FOR PSLD
# it = 0
# for item in os.listdir(args.dir_path):
#     # Check if the item is a directory (folder)
#     list_imgs = []
#     if item.isnumeric() and item < '00101':
#         for img_id in os.listdir(os.path.join(args.dir_path, item)):

#                 # print(item)
#                 list_imgs.append(img_id)
#                 # Remove element that has (img_id.split('_')[2]).split('.')[0] != 'grid' in its name from list_imgs
#         print(list_imgs)

#         list_imgs = [x for x in list_imgs if x.split('_')[0] == 'temp']
#         idx_img = np.random.choice(len(list_imgs))
#         img_true = Image.open(os.path.join(args.dir_path, item, list_imgs[idx_img])).convert('RGB')

#         img_torch = transforms.ToTensor()(img_true)

#         # Create folder if does not exist
#         if not os.path.exists(dest_folder):
#             os.makedirs(dest_folder)

#         save_image(img_torch, f'{dest_folder}/{item}.png')
#     it = it + 1

# FOR REDDIFF - 3 SAMPLES
# it = 0
# for item in os.listdir(args.dir_path):
#     # Check if the item is a directory (folder)
#     list_imgs = []
#     if item.isnumeric() and item < '00101':
#         for img_id in os.listdir(os.path.join(args.dir_path, item)):

#                 # print(item)
#                 list_imgs.append(img_id)
#                 # Remove element that has (img_id.split('_')[2]).split('.')[0] != 'grid' in its name from list_imgs
#         print(list_imgs)

#         # print(item)
#         # list_imgs = [x for x in list_imgs if (x.split('_')[2]).split('.')[0] != 'grid']
#         idx_img = random.sample([0, 1, 2, 3], 3)
#         # print(idx_img)
#         # idx_img = np.random.choice(len(list_imgs))
#         for ii in range(3):
#             img_true = Image.open(os.path.join(args.dir_path, item, list_imgs[idx_img[ii]])).convert('RGB')

#             img_torch = transforms.ToTensor()(img_true)

#             # Create folder if does not exist
#             dest_folder = args.dest_folder + f'/{item}'
#             if not os.path.exists(dest_folder):
#                 os.makedirs(dest_folder)

#             save_image(img_torch, f'{dest_folder}/x_{ii+1}.png')
#     it = it + 1


# Resize
# it = 0
# list_imgs = os.listdir(args.dir_path)
# for item in list_imgs:
#     # Check if the item is a directory (folder)
#     # list_imgs = os.listdir(args.dir_path)
#     if item.split('.')[1] == 'csv':
#         continue
#     img_true = Image.open(os.path.join(args.dir_path, item)).convert('RGB')

#     img_torch = transforms.ToTensor()(img_true)
#     # Resize to 512x512
#     img_torch = transforms.Resize((512, 512))(img_torch)
#     # upsample = nn.Upsample(scale_factor=2, mode='nearest')
#     # img_torch = upsample(img_torch.unsqueeze(0)).squeeze()

#     # # Create folder if does not exist
#     if not os.path.exists(dest_folder):
#         os.makedirs(dest_folder)

#     save_image(img_torch, f'_exp/input/IMAGENET_512/{item}')
#     it = it + 1
