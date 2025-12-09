"""Implementations of various probability distributions."""
import math

import torch

LOG_2PI = math.log(2 * math.pi)


def gaussian_diag_logprob(x, mu, logvar):
    """
    Computes the log probability of a Gaussian distribution with diagonal covariance.
    """
    var = torch.exp(logvar)
    assert (
        x.shape == mu.shape == var.shape
    ), f"Shapes do not match: x {x.shape}, mu {mu.shape}, var {var.shape}"

    return -0.5 * torch.sum(
        (x - mu).pow(2) * torch.exp(-logvar) + logvar + LOG_2PI, dim=-1
    )


def gaussian_std_logprob(x):
    """
    Standard Gaussian log probability.
    """
    return -0.5 * (x.pow(2) + LOG_2PI)


def gaussian_iso_logprob(x, mu, logvar, sequential=False):
    """
    Computes the log probability of a Gaussian distribution with isotropic covariance.
    logvar is a float.
    """
    d = x.shape[-1]
    def single_ver(x, mu, logvar):
        return -0.5 * (
            torch.sum((x - mu).pow(2) * math.exp(-logvar), dim=-1) + d * (logvar + LOG_2PI)
        )
    if sequential:
        bs = x.shape[0]
        out_list = []
        for i in range(bs):
            for j in range(bs):
                out_list.append(single_ver(x[i], mu[j], logvar))
        return torch.stack(out_list, dim=0).reshape(bs, bs)
    else:
        return single_ver(x, mu, logvar)
