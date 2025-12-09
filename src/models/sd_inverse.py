import torch

from diffusers import StableDiffusionPipeline

class SDInversePipeline(StableDiffusionPipeline):
    def decode_latents(self, latents, stay_on_device=False):
        latents = 1 / self.vae.config.scaling_factor * latents
        self.vae.train()
        # set self.vae to require grad
        for param in self.vae.parameters():
            param.requires_grad = True
        image = self.vae.decode(latents).sample
        if not stay_on_device:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def encode_images(self, images):
        with torch.no_grad():
            latent_dist = self.vae.encode(images).latent_dist
            # latents = latent_dist.mean
            latents = latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            # do not scale and treat scaling as part of the decoder
        return latents
