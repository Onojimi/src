import sys
import os
import numpy as np
from optparse import OptionParser
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet.unet_model import Unet
from utils.load import get_ids, split_ids, get_imgs_and_masks
from utils.utils import split_train_val, batch, normalize, resize_and_crop,\
    hwc_to_chw

from torchvision import transforms

def predict_img(net,
                full_img,
                scale_factor = 1,
                out_threshold = 0.5,
                use_dense_crf = False,
                use_gpu = True):
    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]
    
    img = resize_and_crop(full_img,scale = scale_factor)
    img = normalize(img)
    img = hwc_to_chw(img)
    img.torch.from_numpy()
    
    if use_gpu:
        img = img.cuda()
    
    with torch.no_grad():
        img_probs = net(img)
        img_probs = img_probs.squeeze(0)
        
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        
        img_probs = tf(img)
        img_mask_np = img_probs.squeeze().cpu().numpy()
    
#     if use_dense_crf:
#         img_mask = dense_crf(np.array(full_img))
    return img_mask_np > out_threshold
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()     

   
    