import os
import sys
import argparse
from multiprocessing import freeze_support
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (
    ResidualDiffusion, Trainer, Unet, UnetRes, set_seed
)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="SSIM and PSNR Test Runner")

    parser.add_argument('--device', type=str, default='0,1,2,3', help='CUDA device id')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--original_ddim_ddpm', action='store_true', help='Use original DDIM/DDPM')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--num_unet', type=int, default=2)
    parser.add_argument('--objective', type=str, default='pred_res_noise')
    parser.add_argument('--test_res_or_noise', type=str, default='res_noise')
    parser.add_argument('--model_ckpt', type=str,default='/mnt/data/result_ge47nej/results_translation_train/sample_50_epochs_512_imagesize/model-43.pt',
                                                         choices=['/mnt/data/result_ge47nej/results_translation_train/pred_res_noise_ssim/model-10.pt',
                                                                  '/mnt/data/result_ge47nej/results_translation_train/sample_50_epochs_512_imagesize/model-43.pt',
                                                                  '/mnt/data/result_ge47nej/results_translation_train/sample_50_epochs_512_imagesize/model-44.pt'])
    parser.add_argument('--results_folder', type=str,default='/mnt/data/ge54xof/inf_imgs/ssim_loss/')
    parser.add_argument('--heatmap_path', type=str, default=None)

    ### wandb setup ###
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging, default False for test')
    parser.add_argument('--wandb_exp_name', type=str, default='IHC_inference',required=False)
    return parser.parse_args()

def run_test(rank, world_size, args):
    setup(rank, world_size)
    set_seed(10)

    condition = not args.original_ddim_ddpm
    input_condition = False
    input_condition_mask = False
    img_to_img_translation = condition and not input_condition

    folder = [
        "/mnt/data/BCI/train/IHC",
        "/mnt/data/BCI/train/HE",
        "/mnt/data/BCI/test/IHC",
        "/mnt/data/BCI/test/HE"
    ]

    if args.original_ddim_ddpm:
        model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
        diffusion = GaussianDiffusion(
            model,
            image_size=args.image_size,
            timesteps=1000,
            sampling_timesteps=250,
            loss_type='l1'
        )
    else:
        model = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            num_unet=args.num_unet,
            condition=condition,
            input_condition=input_condition,
            objective=args.objective,
            test_res_or_noise=args.test_res_or_noise,
            img_to_img_translation=img_to_img_translation
        )
        diffusion = ResidualDiffusion(
            model,
            image_size=args.image_size,
            timesteps=1000,
            num_samples=args.num_samples,
            sampling_timesteps=10,
            objective=args.objective,
            loss_type='l2',
            condition=condition,
            sum_scale=1,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
            test_res_or_noise=args.test_res_or_noise,
            img_to_img_translation=img_to_img_translation
        )

    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=args.train_batch_size,
        num_samples=args.num_samples,
        train_lr=2e-4,
        train_num_steps=300,
        gradient_accumulate_every=8,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        condition=condition,
        save_and_sample_every=5,
        equalizeHist=False,
        crop_patch=False,
        generation=True,
        num_unet=args.num_unet,
    )

    trainer.device = torch.device(f"cuda:{rank}")
    trainer.load(args.model_ckpt)

    trainer.test_dist(
        result_folder_heat_noise=args.heatmap_path,
        result_folder_sample=args.results_folder,
        last=False,
        FID=False,
        XAI=False
    )

    cleanup()

def main():
    args = parse_args()
    freeze_support()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    world_size = len(args.device.split(','))
    print(f"Detected {world_size} GPU(s).")

    if world_size > 1:
        mp.spawn(run_test, args=(world_size, args), nprocs=world_size, join=True)
    else:
        set_seed(10)

        # fallback to single-GPU logic
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        condition = not args.original_ddim_ddpm
        input_condition = False
        input_condition_mask = False
        img_to_img_translation = condition and not input_condition

        folder = [
            "/mnt/data/BCI/train/IHC",
            "/mnt/data/BCI/train/HE",
            "/mnt/data/BCI/test/IHC",
            "/mnt/data/BCI/test/HE"
        ]

        if args.original_ddim_ddpm:
            model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
            diffusion = GaussianDiffusion(
                model,
                image_size=args.image_size,
                timesteps=1000,
                sampling_timesteps=250,
                loss_type='l1'
            )
        else:
            model = UnetRes(
                dim=64,
                dim_mults=(1, 2, 4, 8),
                num_unet=args.num_unet,
                condition=condition,
                input_condition=input_condition,
                objective=args.objective,
                test_res_or_noise=args.test_res_or_noise,
                img_to_img_translation=img_to_img_translation
            )
            diffusion = ResidualDiffusion(
                model,
                image_size=args.image_size,
                timesteps=1000,
                num_samples=args.num_samples,
                sampling_timesteps=10,
                objective=args.objective,
                loss_type='l2',
                condition=condition,
                sum_scale=1,
                input_condition=input_condition,
                input_condition_mask=input_condition_mask,
                test_res_or_noise=args.test_res_or_noise,
                img_to_img_translation=img_to_img_translation
            )

        trainer = Trainer(
            diffusion,
            folder,
            train_batch_size=args.train_batch_size,
            num_samples=args.num_samples,
            train_lr=2e-4,
            train_num_steps=300,
            gradient_accumulate_every=8,
            ema_decay=0.995,
            amp=False,
            convert_image_to="RGB",
            condition=condition,
            save_and_sample_every=5,
            equalizeHist=False,
            crop_patch=False,
            generation=True,
            num_unet=args.num_unet,
        )

        trainer.load(args.model_ckpt)

        trainer.test(
            result_folder_heat_noise=args.heatmap_path,
            result_folder_sample=args.results_folder,
            last=False,
            FID=False,
            XAI=False
        )

if __name__ == '__main__':
    main()
