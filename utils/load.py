# -*- coding: gb2312 -*-

import os
import numpy as np
from PIL import Image

from utils.utils import resize_and_crop, normalize, hwc_to_chw



def get_ids(img_dir):
#       读取当前文件夹下的所有文件名
    return (f[:-4] for f in os.listdir(img_dir))

def to_cropped_imgs(ids, img_dir, suffix, scale):
    for id in ids:
        #dor id, pos in ids: pos指的是元组 中表示位置的0/1
        img = resize_and_crop(Image.open(img_dir + id + suffix), scale)
        # img是一个np_array. 原来这里是yield square(img,pos)
        yield img
        
def get_imgs_and_masks(ids, img_dir, mask_dir, scale):
    imgs =  to_cropped_imgs(ids, img_dir, '.tif', scale)
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    
    masks = to_cropped_imgs(ids, mask_dir, "_mask.png", scale)
    return zip(imgs_normalized, masks)

def get_full_img_and_mask(id, img_dir, mask_dir):
    img_pil = Image.open(img_dir + id + '.tif')
    mask_pil = Image.open(mask_dir + id + '_mask.png')
    return np.array(img_pil), np.array(mask_pil)