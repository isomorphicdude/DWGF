"""A Wasserstein gradient flow approach."""

import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm, trange

from models.sd_inverse import SDInversePipeline
from utils.distributions import gaussian_iso_logprob
from utils.torch_utils import get_logger
from .rsd import RSD


class DWGF(RSD):
    def __init__(self, model: SDInversePipeline, cfg: DictConfig):
        super().__init__(model, cfg)
        self.model = model
        self.logger = get_logger(__name__, cfg=cfg)

    def sample(self, x, y, ts, generator, y_0, results256=False, **unusedkwargs):
        # here batch size is the number of particles
        # we assume only a single image is processed at a time
        self.ts = ts
        if len(x.shape) == 3:
            batch_size = 1
        else:
            batch_size = x.size(0)
        H = self.H
        input_img = x
        assert self.prompt_embeds is not None, "Prompt embeddings cannot be None"
        latents = self.model.prepare_latents(
            batch_size,
            self.num_channels_latents,
            self.height,
            self.width,
            dtype=self.prompt_embeds.dtype,
            device=self.device,
            generator=generator,
        )

        # optimize latents
        latents = torch.autograd.Variable(latents, requires_grad=True)
        optimizer_z = torch.optim.Adam(
            [latents], lr=self.lr_z, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0
        )

        counter = 0
        x_list = []
        sigmas = torch.sqrt(
            (1 - self.model.scheduler.alphas_cumprod)
            / self.model.scheduler.alphas_cumprod
        )
        sigmas = sigmas[self.model.scheduler.timesteps.cpu().numpy()]  # invert

        # diffusion loop
        for i, t in enumerate(ts[:-1]):
            self.logger.info(f"Step {i+1}/{len(ts)-1}, t={t.item()}")
            noise_t = torch.randn_like(latents).cuda()
            alpha_t = self.model.scheduler.alphas_cumprod[
                self.model.scheduler.timesteps[i].cpu().numpy()
            ]
            # noisy latents at time t
            z_t = self.model.scheduler.add_noise(latents, noise_t, t)

            # computing the prior regularization term
            with torch.no_grad():
                et = self.model.unet(
                    z_t,
                    t,
                    encoder_hidden_states=self.prompt_embeds,
                    cross_attention_kwargs=None,
                ).sample
                # score from noise pred
                score_t = -1 * et / (1 - alpha_t).sqrt()

            # Weighting
            snr_inv = (1 - alpha_t).sqrt() / alpha_t.sqrt()

            z_t_grad = torch.autograd.Variable(z_t, requires_grad=True)
            forward_ker = gaussian_iso_logprob(
                z_t_grad.view(batch_size, -1),
                (latents * alpha_t.sqrt()).detach().view(batch_size, -1),
                math.log(1 - alpha_t),
                sequential=True,  # save memory
            )

            # size (bs, bs) -> (bs,)
            forward_ker = torch.logsumexp(forward_ker, dim=1) - math.log(batch_size)

            forward_score = torch.autograd.grad(forward_ker.sum(), z_t_grad)[0]

            prior_loss = (
                (
                    torch.mul(forward_score.detach(), latents).mean()
                    - torch.mul(score_t.detach(), latents).mean()
                )
                * snr_inv
                * self.w_t
                * (1 - alpha_t).sqrt()
            )

            # data likelihood term
            x_pred_z = self.model.decode_latents(latents, stay_on_device=True)
            noise = torch.randn_like(x_pred_z) * self.cfg.algo.decoder_std
            u_loss_obs = ((y_0 - H.H(x_pred_z + noise)) ** 2).mean() / 2

            with torch.no_grad():
                x_decode_encode = self.model.decode_latents(
                    self.model.encode_images(x_pred_z + noise), stay_on_device=True
                )
                cotan_prior = -1 * (x_decode_encode - x_pred_z).detach()

            vjp_loss = (torch.mul(x_pred_z, (cotan_prior).detach())).mean()

            # total loss
            vjp_coeff = self.cfg.algo.vjp_coeff
            loss = u_loss_obs + prior_loss + vjp_loss * vjp_coeff

            # optimize
            optimizer_z.zero_grad()
            loss.backward()
            optimizer_z.step()

            # save
            if self.cfg.exp.save_evolution:
                self.save_evo(counter, x_pred_z, latents, input_img, x_list, results256)
            else:
                x_list.append(x_pred_z.clone().detach())

            counter += 1

        return x_list[-1], x_pred_z.clone().detach()
