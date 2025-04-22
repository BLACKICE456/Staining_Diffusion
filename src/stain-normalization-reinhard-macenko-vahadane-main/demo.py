#%% import
from __future__ import division

from utils import stain_utils as utils
from models import stainNorm_Vahadane, stainNorm_Reinhard, stainNorm_Macenko

import numpy as np
import os
import random
from pathlib import Path

import Augmentor
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
#%% defines
# normalization method (Reinhard [0], Macenko [1], Vahadane [2])
def load_flist(flist):
    my_files = os.listdir(flist)

    path_list = []

    for file in my_files:
        if os.path.isdir(flist + "/" + file):
            continue

        path_list.append(flist + "/" + file)

    return path_list




def evaluate(img1,img2):

    #img1 = img1.squeeze().permute(1,2,0).detach().cpu().numpy()
    #img2 = img2.squeeze().permute(1,2,0).detach().cpu().numpy()

    psnr = peak_signal_noise_ratio(img1,img2)
    ssim = structural_similarity(img1, img2, multichannel=True,channel_axis=-1,data_range=255)
    return psnr,ssim


def image_normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) #/255
    #img = img / 255
    return img

def set_results_folder2(path):
    results_folder = Path(path)
    if not results_folder.exists():
        os.makedirs(results_folder)

#%% load data
if __name__ == '__main__':
    for i in range(0,1):
        normalization_method = i

        if normalization_method == 0:
            # Reinhard
            method = 'Reinhard'
            normalizer = stainNorm_Reinhard.Normalizer()
        elif normalization_method == 1:
            # Macenko
            method = 'Macenko'
            normalizer = stainNorm_Macenko.Normalizer()
        elif normalization_method == 2:
            # Vahadane
            method = 'Vahadane'
            normalizer = stainNorm_Vahadane.Normalizer()
        else:
            print('enter valid normalization method (Reinhard [0], Macenko [1], Vahadane [2])')
            exit()

        input_dir = '/mnt/data/BCI/test/IHC'
        output_dir = '/mnt/data/result_ge47nej/results_translation_test/1024_imgsize_test_res_normalized_417_' + method +'/'
        path_list = load_flist(input_dir)
        img_source_path = '/mnt/data/result_ge47nej/results_translation_test'
        psnr_list = []
        ssim_list = []
        psnr_org_list= []
        ssim_org_list = []
        #path_list = ['/mnt/data/BCI/test/IHC/00959_test_3+.png','/mnt/data/BCI/test/IHC/00407_test_3+.png','/mnt/data/BCI/test/IHC/00378_test_3+.png','/mnt/data/BCI/test/IHC/00007_test_1+.png']

        set_results_folder2(output_dir)
        for file in path_list:
            img_gt = utils.read_image(file)
            file_name = file.split('/')[-1]

            target_name = '/mnt/data/result_ge47nej/results_translation_test/512_imagesize_SFS_sample_checkpoint_44_1024_ablation_res/' + file_name
            #target_name = '/mnt/data/result_ge47nej/web_dir1024_BCI2/images' + '/' + file_name[0:-4]  + '_fake_B.png'
            img_target = utils.read_image(target_name)
            print(img_source_path + '/' + file_name)

            img_source_1 = utils.read_image(img_source_path + '/' + file_name)
            #img_target = cv2.resize(img_target, (512, 512))
            #img_target = image_normalize(img_target)
            #img_source_1 = image_normalize(img_source_1) #compare the ssim and psnr of single image
            normalizer.fit(img_target)
            normalized_img = normalizer.transform(img_source_1)

            #img_target = image_normalize(img_target)
            #normalized_img = image_normalize(normalized_img)

            psnr_org,ssim_org = evaluate(img_gt,img_source_1)
            psnr, ssim = evaluate(img_gt, normalized_img)
           # print(img_target)
           # print(normalized_img)
            #img_target = (img_target*255).astype(np.uint8)
            #img_source_1 = (img_source_1*255).astype(np.uint8)
            #normalized_img = (normalized_img*255).astype(np.uint8)
            save_img =  normalized_img
            #save_img = np.concatenate((img_source_1,img_target,normalized_img),axis=1)
            # Convert the NumPy array to an image
            """
            img_target = Image.fromarray(img_target)
            img_source_1 = Image.fromarray(img_source_1)
            img_target.save(output_dir + 'target_' + file_name)
            img_source_1.save(output_dir + 'source_' + file_name)
            """
            save_img = Image.fromarray(save_img)
            save_img.save(output_dir + 'target_source_normalized_' + file_name)
            print(output_dir + 'target_source_normalized_' + file_name)

            print(file_name)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            psnr_org_list.append(psnr_org)
            ssim_org_list.append(ssim_org)
            print("method:{},psnr:{},ssim:{}, psnr_org:{},ssim_org:{}".format(method,psnr,ssim,psnr_org,ssim_org))


        psnr_mean = np.mean(psnr_list)
        psnr_std = np.std(psnr_list)
        ssim_mean = np.mean(ssim_list)
        ssim_std = np.std(ssim_list)
        psnr_org_mean = np.mean(psnr_org_list)
        psnr_org_std = np.std(psnr_org_list)
        ssim_org_mean = np.mean(ssim_org_list)
        ssim_org_std = np.std(ssim_org_list)

        print("method:{},psnr_mean:{},psnr_std:{}".format(method,psnr_mean, psnr_std))
        print("method:{},ssim_mean:{},ssim_std_{}".format(method,ssim_mean, ssim_std))
        print("method:{},psnr_org_mean:{},psnr_org_std:{}".format(method,psnr_org_mean, psnr_org_std))
        print("method:{},ssim_org_mean:{},ssim_org_std_{}".format(method,ssim_org_mean, ssim_org_std))





