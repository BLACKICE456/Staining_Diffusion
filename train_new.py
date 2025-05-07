import os
import sys
import argparse
from multiprocessing import freeze_support

import torch
import wandb

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (
    ResidualDiffusion, Trainer, Unet, UnetRes, set_seed
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RDDM training")

    # General configuration
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--original_ddpm', action='store_true', help='Use original DDIM/DDPM mode')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use (e.g., "0" or "0,1")')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--steps', type=int, default=None, help='Override train steps (optional)')
    parser.add_argument('--pretrained_path', type=str, default='/mnt/data/result_ge47nej/results_translation_train/sample_50_epochs_512_imagesize/model-20.pt', help='Path to pretrained model')

    return parser.parse_args()


def get_config(args):
    if args.debug:
        return {
            'save_and_sample_every': 2,
            'sampling_timesteps': 10,
            'sampling_timesteps_original': 10,
            'train_num_steps': args.steps or 10
        }
    return {
        'save_and_sample_every': 5,
        'sampling_timesteps': 10,
        'sampling_timesteps_original': 250,
        'train_num_steps': args.steps or 300
    }


def get_data_folders(condition, input_condition):
    if not condition:
        return []

    if input_condition:
        return [
            "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
            "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
            "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_train.flist",
            "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
            "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist",
            "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_test.flist"
        ]
    else:
        return [
            "/mnt/data/BCI/train/IHC",
            "/mnt/data/BCI/train/HE",
            "/mnt/data/BCI/test/IHC",
            "/mnt/data/BCI/test/HE"
        ]


def create_model(args, config):
    image_size = 1024
    num_unet = 2
    objective = 'pred_res_noise'
    test_res_or_noise = "res_noise"
    condition = not args.original_ddpm
    input_condition = False
    input_condition_mask = False
    img_to_img_translation = condition and not input_condition
    sum_scale = 1
    num_samples = 2

    if args.original_ddpm:
        model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
        diffusion = GaussianDiffusion(
            model,
            image_size=image_size,
            timesteps=1000,
            sampling_timesteps=config['sampling_timesteps_original'],
            loss_type='l1',
        )
    else:
        model = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            num_unet=num_unet,
            condition=condition,
            input_condition=input_condition,
            objective=objective,
            test_res_or_noise=test_res_or_noise,
            img_to_img_translation=img_to_img_translation
        )
        diffusion = ResidualDiffusion(
            model,
            image_size=image_size,
            timesteps=1000,
            num_samples=num_samples,
            sampling_timesteps=config['sampling_timesteps'],
            objective=objective,
            loss_type='l2',
            condition=condition,
            sum_scale=sum_scale,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
            test_res_or_noise=test_res_or_noise,
            img_to_img_translation=img_to_img_translation
        )

    return diffusion, {
        'folder': get_data_folders(condition, input_condition),
        'train_batch_size': 2,
        'num_samples': num_samples,
        'train_lr': 2e-4,
        'train_num_steps': config['train_num_steps'],
        'gradient_accumulate_every': 8,
        'ema_decay': 0.995,
        'amp': False,
        'convert_image_to': "RGB",
        'condition': condition,
        'save_and_sample_every': config['save_and_sample_every'],
        'equalizeHist': False,
        'crop_patch': False,
        'generation': True,
        'num_unet': num_unet
    }


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.stdout.flush()
    set_seed(args.seed)

    config = get_config(args)
    diffusion, trainer_kwargs = create_model(args, config)
    trainer = Trainer(diffusion, **trainer_kwargs)

    wandb_project = "RDDM_"
    log = wandb.init(project=wandb_project, resume=False)
    log.config.update({
        'steps': config['train_num_steps'],
        'scale': 1,
        'train_batch': 2
    })

    if args.pretrained_path:
        trainer.load(args.pretrained_path)

    trainer.train(log_obj=log)
    log.finish()


if __name__ == '__main__':
    freeze_support()
    main()
