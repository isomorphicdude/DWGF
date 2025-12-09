# -----------------------------------------------------------
# Ported from IPLD
# This file contains the computation code for FID using `torchmetrics` package.
# See https://torchmetrics.readthedocs.io/en/stable/ for more details.
# -----------------------------------------------------------

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

def compute_fid(dataset_samples: torch.Tensor,
                model_samples: torch.Tensor,
                device: str = 'cuda',
                batch_size: int = 10):
    """
    Computes the Fréchet Inception Distance (FID) between two sets of images,
    which can have different numbers of samples.
    """
    # convert to [0, 255] and uint8, as required by torchmetrics
    real_images = (dataset_samples * 255).to(torch.uint8)
    fake_images = (model_samples * 255).to(torch.uint8)

    # repeat channels to be RGB
    if real_images.shape[1] == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
    if fake_images.shape[1] == 1:
        fake_images = fake_images.repeat(1, 3, 1, 1)

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    n_real = real_images.shape[0]
    print(f"Processing {n_real} real images...")
    for i in range(0, n_real, batch_size):
        real_batch = real_images[i:i+batch_size, ...].to(device)
        fid_metric.update(real_batch, real=True)

    n_fake = fake_images.shape[0]
    print(f"Processing {n_fake} fake images...")
    for i in range(0, n_fake, batch_size):
        fake_batch = fake_images[i:i+batch_size, ...].to(device)
        fid_metric.update(fake_batch, real=False)

    print("Computing FID score...")
    fid_value = fid_metric.compute()
    return fid_value.item()
