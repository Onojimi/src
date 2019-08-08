# -*- coding: gb2312 -*-

import random
import numpy as np


def get_square(img_np, pos):
    '''����ԭ��ͼƬ�ĸ߽س�һ��������.����ԭ�����ݼ���ͼƬ�ǳ����ε�'''
    h = img_np.shape[0]
    if pos == 0:
        return img_np[:,:h]
    else:
        return img_np[:,-h:]

def split_img_into_sqares(img_np):
    '''����ԭ��ͼƬ�ĸ߽س�һ��������.'''
    return get_square(img_np, 0), get_square(img_np, 0)

def hwc_to_chw(img_np):
    '''img_pilת����img_np��ά����(h,w,c,)��ת����(c,h,w)'''
    return np.transpose(img_np, axes = [2,0,1])

def resize_and_crop(img_pil, scale = 0.5, final_height = None):
    '''����СͼƬ��С,�����Ҫ�Ļ��ٽ�һ���ü�ͼƬ'''
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
    #����batch
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if(i + 1)%batch_size == 0:
            yield b
            b = []
            
    if len(b)>0:
        yield b

def split_train_val(dataset,val_percent = 0.1):
    #�������ݼ�,ֻ�ǵ����Ĵ������ϻ���
    #����һ���ֵ�,����key�ֱ���train,val
    #value���ļ���
    dataset = list(dataset)
    length = len(dataset)
    n = int(length*val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n],'val':dataset[-n:]}

def normalize(x):
    #����ÿ�����ص��brightness[0,255]��������
    return x/255

def merge_masks(img1, img2, full_w):
    #�Գ����ε�ͼƬ���������һ��mask�ұ�һ��mask���ٰ�����mask�ں�����
    h = img1.shape[0]
    
    new = np.zeros((h,full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1] 
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]
    
    return new
