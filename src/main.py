import time
import os

import torch
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import lpips
import wandb

from custom_datasets import build_loader
from models.sd_inverse import SDInversePipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from utils.degredations import build_degredation_model, get_degreadation_image
from utils.functions import postprocess, preprocess
from utils.torch_utils import get_logger, init_omega, seed_everything, ensure_dir
from utils.img_utils import resize_input
from algos import build_algo
from utils.save import save_result
from eval.compute_fid import compute_fid


def prepare_paths(cfg):
    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, cfg.algo.name_folder)
    output_path = os.path.join(cfg.exp.root, cfg.exp.output_path, cfg.algo.name_folder)
    evol_path = os.path.join(cfg.exp.root, cfg.exp.evol_path, cfg.algo.name_folder)
    for path in (deg_path, output_path, evol_path):
        ensure_dir(path)
    return deg_path, output_path, evol_path


def build_pipeline(cfg):
    pipe = SDInversePipeline.from_pretrained("Manojb/stable-diffusion-2-1-base")
    pipe = pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(cfg.exp.num_steps, device="cuda")
    return pipe, pipe.scheduler.timesteps


def load_dino(cfg):
    # only for repulsive score distillation
    if cfg.algo.name in ["rsd_stable", "rsd_stable_nonaug"]:
        return torch.hub.load("facebookresearch/dino:main", "dino_vits16").to("cuda")
    return None


def expand_particles(x, n_particles):
    if n_particles > 1:
        return x.repeat(n_particles, 1, 1, 1)
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x


def build_measurement(x, H, sigma):
    y_0 = H.H(x.to("cuda"))
    return y_0 + torch.randn_like(y_0) * sigma


def save_degraded_(cfg, y_0, H, logger):
    if cfg.exp.save_deg is not True:
        return
    if cfg.algo.deg == "deblur_motion":
        xo = postprocess(y_0)
    else:
        xo = postprocess(get_degreadation_image(y_0, H, cfg))
    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, "x_deg.png")
    save_image(xo[0].cpu(), deg_path)
    logger.info(f"Degradated image y saved in {deg_path}")


def sample_with_algo(cfg, algo, x, y, ts, generator, y_0, dino):
    if cfg.algo.name in ["rsd_stable", "rsd_stable_nonaug"]:
        return algo.sample(x, y, ts, generator, y_0, dino=dino)
    return algo.sample(x, y, ts, generator, y_0)


def compute_psnr(xo, x_gt):
    mse = torch.mean((xo - x_gt.cpu()) ** 2, dim=(1, 2, 3))
    return 10 * torch.log10(1 / (mse + 1e-10))


def compute_lpips_score(lpips_model, xo, x_gt, move_ref_to_cuda=True):
    reference = x_gt.cuda() if move_ref_to_cuda else x_gt
    return lpips_model(xo.cuda(), reference)


def process_single_image(cfg, algo, H, ts, generator, dino, lpips_, output_path, logger):
    img_path = os.path.join(cfg.exp.img_path, cfg.exp.img_id)
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).cuda()

    logger.info(f"Input image shape: {x.shape}, resizing to 512x512...")
    x = resize_input(x)
    x = expand_particles(x, cfg.algo.n_particles)

    y = None
    x = preprocess(x)
    y_0 = build_measurement(x, H, cfg.algo.sigma_x0)
    logger.info(f"Images loaded and measurements generated")

    save_degraded_(cfg, y_0, H, logger)

    start = time.time()
    xt_s, _ = sample_with_algo(cfg, algo, x, y, ts, generator, y_0, dino)
    end = time.time()
    print(f"Time taken: {end - start}")

    if isinstance(xt_s, list):
        xo = postprocess(xt_s[0]).cpu()
        x_gt = postprocess(x)
    else:
        xo = postprocess(xt_s).cpu()
        x_gt = postprocess(x)

    psnr = compute_psnr(xo, x_gt)
    logger.info(f"Mean PSNR: {psnr.mean()}")
    logger.info(f"PSNR: {psnr}")

    LPIPS = compute_lpips_score(lpips_, xo, x_gt)
    logger.info(f"Mean LPIPS: {LPIPS.mean()}")
    logger.info(f"LPIPS: {LPIPS[:,0,0,0]}")

    img_id = cfg.exp.img_id.split(".")[0]
    output_path_img = f"{output_path}/{img_id}"
    ensure_dir(output_path_img)

    for i in range(cfg.algo.n_particles):
        save_image(xo[i], f"{output_path_img}/x_hat_{i}.png")
    image_grid = make_grid(xo.cpu())
    save_image(image_grid, f"{output_path_img}/x_hat_grid.png")

    torch.cuda.empty_cache()
    logger.info(f"Done. You can fine the generated images in {output_path_img}")


def process_dataset(cfg, algo, H, ts, generator, dino, lpips_, output_path, logger, dataset_name, max_num_images):
    loader = build_loader(cfg)
    logger.info(f"Dataset size is {len(loader.dataset)}")

    output_path_deg = os.path.join(output_path, cfg.algo.deg)
    ensure_dir(output_path_deg)

    psnr_list = []
    lpips_list = []

    for it, (x, y, info) in enumerate(loader):
        if it >= max_num_images:
            logger.info(f"Reached max number of images: {max_num_images}")
            break
        logger.info(f"Input image:{ info['index'][0]}")

        x = resize_input(x)
        x = x.cuda()
        y = y.cuda()
        x = preprocess(x)
        kwargs = info

        x = x.repeat(cfg.algo.n_particles, 1, 1, 1)

        y_0 = build_measurement(x, H, cfg.algo.sigma_x0)
        logger.info(f"Images loaded and measurements generated")

        save_degraded_(cfg, y_0, H, logger)

        xt_s, _ = sample_with_algo(cfg, algo, x, y, ts, generator, y_0, dino)

        if isinstance(xt_s, list):
            xo = postprocess(xt_s[0]).cpu()
            x_gt = postprocess(x)
        else:
            xo = postprocess(xt_s).cpu()
            x_gt = postprocess(x)

        save_result(dataset_name, xo, y, info, output_path_deg, "")

        psnr = compute_psnr(xo, x_gt)
        print(f"Mean PSNR: {psnr.mean()}")
        print(f"PSNR: {psnr}")
        psnr_list.append(psnr.mean().item())

        LPIPS = compute_lpips_score(lpips_, xo, x_gt, move_ref_to_cuda=False)
        print(f"Mean LPIPS: {LPIPS.mean()}")
        print(f"LPIPS: {LPIPS[:,0,0,0]}")
        lpips_list.append(LPIPS.mean().item())

        if cfg.exp.use_wandb:
            wandb.log(
                {
                    "psnr": psnr.mean().item(),
                    "lpips": LPIPS.mean().item(),
                },
                step=it,
            )

        torch.cuda.empty_cache()
        logger.info(f"Done. You can fine the generated images in {output_path_deg}")

    return psnr_list, lpips_list


@hydra.main(
    version_base="1.2", config_path="../configs", config_name="ffhq_stable_diffusion"
)
def main(cfg: DictConfig):
    seed_everything(cfg.exp.seed)
    generator = torch.Generator(device="cuda").manual_seed(cfg.exp.seed)

    # Load config file
    cwd = HydraConfig.get().runtime.output_dir
    cfg = init_omega(cfg, cwd)

    # Set folder name
    cfg.algo.name_folder = cfg.algo.name

    # Build paths
    deg_path, output_path, evol_path = prepare_paths(cfg)
    dataset_name = cfg.dataset.name

    # Get logger
    logger = get_logger(name="main", cfg=cfg)
    if cfg.exp.use_wandb:
        wandb.init(
            project=cfg.exp.get("name", "GradientFlowInverse"),
            name=f"{cfg.algo.name_folder}_{cfg.algo.deg}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    pipe, ts = build_pipeline(cfg)
    logger.info(f"Model loaded")

    dino = load_dino(cfg)
    lpips_ = lpips.LPIPS(net="alex").cuda()  # best forward score

    # Load degradation model
    algo = build_algo(pipe, cfg)
    H = algo.H
    logger.info(
        f"Degradation model loaded. The experiment corresponds to {cfg.algo.deg}"
    )

    ### UP TO THIS POINT, EVERYTHING IS THE SAME EITHER FOR STABLE DIFFUSION OR OTHER DIFFUSION MODELS.
    max_num_images = cfg.exp.max_num_images
    psnr_list = []
    lpips_list = []
    if cfg.algo.name in ["rsd_stable", "rsd_stable_nonaug", "dwgf"]:
        if cfg.exp.load_img_id is True:
            process_single_image(
                cfg, algo, H, ts, generator, dino, lpips_, output_path, logger
            )
        else:
            psnr_list, lpips_list = process_dataset(
                cfg,
                algo,
                H,
                ts,
                generator,
                dino,
                lpips_,
                output_path,
                logger,
                dataset_name,
                max_num_images,
            )
    else:
        raise NotImplementedError("Wrong method")
    # compute FID
    if not cfg.exp.load_img_id:
        fid_score = compute_fid(cfg)
        logger.info(f"FID score: {fid_score}")
        # log results
        if cfg.exp.use_wandb:
            wandb.log(
                {
                    "psnr": sum(psnr_list) / len(psnr_list),
                    "lpips": sum(lpips_list) / len(lpips_list),
                    "fid": fid_score,
                }
            )
        logger.info(f"Mean PSNR: {sum(psnr_list) / len(psnr_list)}")
        logger.info(f"Mean LPIPS: {sum(lpips_list) / len(lpips_list)}")


if __name__ == "__main__":
    main()
