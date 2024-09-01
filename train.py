import os
import sys
from multiprocessing import freeze_support
from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)

# init
#os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0,1,2,3])
os.environ['NCCL_IB_TIMEOUT'] = '22'
os.environ['NCCL_IB_GID_INDEX'] = '3'
sys.stdout.flush()
set_seed(10)
debug = False



if debug:
    save_and_sample_every = 2
    sampling_timesteps = 2
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 20
    resume = False
else:
    save_and_sample_every = 10000
    """
    if len(sys.argv) > 1:
        print(sys.argv)
        sampling_timesteps = int(sys.argv[1])
    else:
    """
    sampling_timesteps = 10 # (50,)
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 100000 #100000
    resume = False
    parallel_test_bool = True

    #define epochs
'''
original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = False
    input_condition = False
    input_condition_mask = False
'''
original_ddim_ddpm = False
condition = 1
input_condition = False
input_condition_mask = False
if condition:
    # Image restoration  


    folder = [r"/mnt/data/BCI/train/IHC",
              r"/mnt/data/BCI/train/HE",
              r"/mnt/data/BCI/test/IHC",
              r"/mnt/data/BCI/test/HE"]

    train_batch_size = 2
    num_samples = 2
    sum_scale = 0.1
    image_size = 512 #check the consume of image and model

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
        test_res_or_noise = test_res_or_noise
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type='l2',            # L1 or L2
        condition=condition,
        sum_scale=sum_scale,
        # scale of noise
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
        test_res_or_noise = test_res_or_noise
    )

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=2e-4,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=True,
    generation=True,
    num_unet=num_unet,
    resume_load = resume,
    parallel_test = parallel_test_bool,
    sum_scale_train=sum_scale
)

# train

if __name__ == '__main__':

    freeze_support()

    if resume and trainer.accelerator.is_local_main_process:
        trainer.load(4)


    
    if not parallel_test_bool:
        trainer.train()

    if parallel_test_bool:
        for i in range(4,5):

            trainer.load(i)
            """
            trainer.set_results_folder(
                '/mnt/data/result_ge47nej/test_timestep_' + str(sampling_timesteps) + "_" + str(train_num_steps) + "num_scale_" + str(sum_scale))
            trainer.set_results_folder_sample(
                '/mnt/data/result_ge47nej/test_timestep_sample_' + str(sampling_timesteps) + " " + str(train_num_steps) + "num_scale_" + str(sum_scale))
            """
            trainer.parallel_test_sample(last=True,ckpt_num=i)

    # test
    """
    if not trainer.accelerator.is_local_main_process:
        pass
    else:
        trainer.load(trainer.train_num_steps//save_and_sample_every)
        trainer.set_results_folder(
            './results/test_timestep_'+str(sampling_timesteps) + str(train_num_steps))
        trainer.set_results_folder_sample(
            './results/test_timestep_sample_' + str(sampling_timesteps) + str(train_num_steps))

        #trainer.test(last=True)
    """

# trainer.set_results_folder('./results/test_sample')
# trainer.test(sample=True)
