import os
import numpy as np
from PIL import Image

def rotate_pic(im, im_name, type):
    im1 = im.transpose(Image.FLIP_LEFT_RIGHT)
    im2 = im1.transpose(Image.FLIP_TOP_BOTTOM)
    im3 = im.transpose(Image.ROTATE_270)
    
    if type is 'images':
        im.save("images_cut/"+im_name+"1.tif")
        im1.save("images_cut/"+im_name+"2.tif")
        im2.save("images_cut/"+im_name+"3.tif")
        im3.save("images_cut/"+im_name+"4.tif")
        
    elif type is 'masks':
        im.save("masks_cut/"+im_name+"1_mask.png")
        im1.save("masks_cut/"+im_name+"2_mask.png")
        im2.save("masks_cut/"+im_name+"3_mask.png")
        im3.save("masks_cut/"+im_name+"4_mask.png")

def cut_pics(im_name):
    img = Image.open("images/"+im_name+".tif")
    mask = Image.open("masks/"+im_name+"_mask.png")
    
    img_np = np.array(img)
    mask_np = np.array(mask)
    
    h = img_np.shape[0]
    w = img_np.shape[1]  
    
    mid_h = int(h/2)
    mid_w = int(w/2)
    
    img_lt = Image.fromarray(img_np[:mid_h,:mid_w], 'RGB')
    rotate_pic(img_lt, im_name+'_1', 'images')
    img_rt = Image.fromarray(img_np[:mid_h,mid_w:], 'RGB')
    rotate_pic(img_rt, im_name+'_2', 'images')
    img_lb = Image.fromarray(img_np[mid_h:,:mid_w], 'RGB')
    rotate_pic(img_lb, im_name+'_3', 'images')
    img_rb = Image.fromarray(img_np[mid_h:,mid_w:], 'RGB')
    rotate_pic(img_rb, im_name+'_4', 'images')
    
    mask_lt = Image.fromarray(mask_np[:mid_h,:mid_w], 'L')
    rotate_pic(mask_lt, im_name+'_1', 'masks')
    mask_rt = Image.fromarray(mask_np[:mid_h,mid_w:], 'L')
    rotate_pic(mask_rt, im_name+'_2', 'masks')
    mask_lb = Image.fromarray(mask_np[mid_h:,:mid_w], 'L')
    rotate_pic(mask_lb, im_name+'_3', 'masks')
    mask_rb = Image.fromarray(mask_np[mid_h:,mid_w:], 'L')
    rotate_pic(mask_rb, im_name+'_4', 'masks')
    
for f in os.listdir("images/"):
    img_name = f[:4]
    cut_pics(img_name)





