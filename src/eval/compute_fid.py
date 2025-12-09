import os
import logging

import torch
from torchvision import transforms
import pytorch_fid_wrapper as pfw
from PIL import Image

from custom_datasets import build_loader
from utils.functions import postprocess, preprocess
from .torchmetrics_fid import compute_fid as compute_fid_torchmetrics

logger = logging.getLogger(__name__)


def compute_fid(cfg):
    loader = build_loader(cfg)
    print(f"length of loader: {len(loader)}")
    real_x = []
    for i, (x, _, info) in enumerate(loader):
        if i >= cfg.exp.max_num_images:
            break
        real_x.append(preprocess(x))
    real_x = torch.cat(real_x, dim=0)
    # real_m, real_s = pfw.get_stats(real_x, device="cuda")

    # read output images
    output_img = get_output_img(cfg)
    if output_img is None:
        logger.warning("No output images found, cannot compute FID.")
        return None
    fake_x = []
    for img_path in output_img:
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img).unsqueeze(0)
        img = preprocess(img)
        fake_x.append(img)
    # do not convert as to tensor converts to [0, 1] range
    fake_x = torch.cat(fake_x, dim=0)
    # fid = pfw.fid(fake_images=fake_x, real_m=real_m, real_s=real_s)
    fid = compute_fid_torchmetrics(real_x, fake_x)
    return fid


def get_output_img(cfg) -> list[str]:
    # folder structure
    # dwgf/inp_random/00000/x_0.png
    # dwgf/inp_random/00000/x_1.png
    output_path = os.path.join(cfg.exp.root, cfg.exp.output_path, cfg.algo.name_folder)
    output_path = os.path.join(output_path, cfg.algo.deg)
    parent_dir = os.listdir(output_path)
    if len(parent_dir) == 0:
        logging.warning(f"No output images found in {output_path}")
        return None
    output_img = []
    for item in parent_dir:
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path):
            for img in os.listdir(item_path):
                # individual images are saved as x_0.png, x_1.png, etc.
                if img.endswith('.png'):
                    output_img.append(os.path.join(item_path, img))
    return output_img
