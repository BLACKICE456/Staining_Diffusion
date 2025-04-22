import os
import sys
import wandb
from multiprocessing import freeze_support

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)
import resnet
# init
#os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [3])

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.stdout.flush()
set_seed(10)
debug = False

if debug:
    save_and_sample_every = 2
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 10
else:
    save_and_sample_every = 5

    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 300
    training = False






original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True
    input_condition = False
    input_condition_mask = False

if condition:
    # Image restoration  
    if input_condition:
        folder = ["xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_test.flist"]
    else:
        """
        folder = [r"/home/ge47nej/RDDM/AFHQ_data/afhq/train/cat",
                  r"/home/ge47nej/RDDM/AFHQ_data/afhq/train/dog",
                  #r"/home/ge47nej/RDDM/AFHQ_data/afhq/val/cat",
                  r"/home/ge47nej/RDDM/AFHQ_data/afhq/val/dog"]
        """
        folder = [r"/mnt/data/BCI/train/IHC",
                  r"/mnt/data/BCI/train/HE",
                  r"/mnt/data/BCI/test/IHC",
                  r"/mnt/data/BCI/test/HE"]
        """
        folder = ["/home/ge47nej/RDDM/ISTD_Dataset/test/test_C",
         "/home/ge47nej/RDDM/ISTD_Dataset/test/test_A",
         "/home/ge47nej/RDDM/ISTD_Dataset/test/test_C",
         "/home/ge47nej/RDDM/ISTD_Dataset/test/test_A"]
         """

        img_to_img_translation = True
    train_batch_size = 2
    num_samples = 2
    sum_scale = 1
    image_size = 1024


num_unet = 2
objective = 'pred_res_noise'
test_res_or_noise = "res_noise"
if original_ddim_ddpm:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        sampling_timesteps=sampling_timesteps_original_ddim_ddpm,
        loss_type='l1',            # L1 or L2
    )
else:
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        input_condition=input_condition,
        objective=objective,
        test_res_or_noise = test_res_or_noise,
        img_to_img_translation = img_to_img_translation
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        num_samples=num_samples,
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type='l2',            # L1 or L2
        condition=condition,
        sum_scale=sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
        test_res_or_noise = test_res_or_noise,
        img_to_img_translation = img_to_img_translation
    )


trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=2e-4,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=8,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=True,
    num_unet=num_unet,
)

# train
if __name__ == '__main__':

    freeze_support()


    project_name = "RDDM_test_draft"
    log_fun = wandb.init(project=project_name,resume = False)
    log_fun.config.update(dict(steps = train_num_steps,scale = sum_scale,train_batch = train_batch_size))


    # test

    for i in range(43,44):

        path = '/mnt/data/result_ge47nej/results_translation_train/sample_50_epochs_512_imagesize' + '/model-' + str(i) +'.pt'

        #print(path)
        trainer.load(path)

        trainer.set_results_folder(
                '/mnt/data/result_ge47nej/results_translation_XAI/imagesize_512_1_23' + str(sampling_timesteps))
        save_heatmap_path ='/mnt/data/result_ge47nej/result_XAI_test/heat_noise_show'
        saver_result_folder_sample = "/mnt/data/result_ge47nej/results_translation_test/512_imagesize_SFS_sample_checkpoint_44_1024_ablation_noise/"
        trainer.test(save_heatmap_path = save_heatmap_path,save_result_folder_sample=saver_result_folder_sample,last=False,XAI=False)

    log_fun.finish()
