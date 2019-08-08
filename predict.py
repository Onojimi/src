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
                use_dense_crf = True,
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
        output = net(img)
        output = output.squeeze(0)
        
    