# -*- coding: gb2312 -*-

import random
import numpy as np


def get_square(img_np, pos):
    '''按照原本图片的高截成一个正方形.这里原本数据集的图片是长方形的'''
    h = img_np.shape[0]
    if pos == 0:
        return img_np[:,:h]
    else:
        return img_np[:,-h:]

def split_img_into_sqares(img_np):
    '''按照原本图片的高截成一个正方形.'''
    return get_square(img_np, 0), get_square(img_np, 0)

def hwc_to_chw(img_np):
    '''img_pil转化成img_np后，维度是(h,w,c,)，转换成(c,h,w)'''
    return np.transpose(img_np, axes = [2,0,1])

def resize_and_crop(img_pil, scale = 0.5, final_height = None):
    '''先缩小图片大小,如果需要的话再进一步裁剪图片'''
    w = img_pil.size[0]
    h = img_pil.size[1]
    w_new = int(w*scale)
    h_new = int(h*scale)
    
    if not final_height:
        diff = 0
    else:
        diff = h_new - final_height
        
    img_pil = img_pil.crop((0, diff//2, w_new, h_new-diff//2))
    return np.array(img_pil, dtype = np.float32)  

def batch(iterable, batch_size):
    #生成batch
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if(i + 1)%batch_size == 0:
            yield b
            b = []
            
    if len(b)>0:
        yield b

def split_train_val(dataset,val_percent = 0.1):
    #划分数据集,只是单纯的从数量上划分
    #返回一个字典,两个key分别是train,val
    #value是文件名
    dataset = list(dataset)
    length = len(dataset)
    n = int(length*val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n],'val':dataset[-n:]}

def normalize(x):
    #对于每个像素点的brightness[0,255]进行正则化
    return x/255

def merge_masks(img1, img2, full_w):
    #对长方形的图片，生成左边一个mask右边一个mask，再把两个mask融合起来
    h = img1.shape[0]
    
    new = np.zeros((h,full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1] 
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]
    
    return new
