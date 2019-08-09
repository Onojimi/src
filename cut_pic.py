import os
import numpy as np
from PIL import Image

def cut_pics(img_name):
    img = Image.open("images/"+img_name+".tif")
    img_np = np.array(img)
    
    h = img_np.shape[0]
    w = img_np.shape[1]  
    
    mid_h = int(h/2)
    mid_w = int(w/2)
    
    img_lt = Image.fromarray(img_np[:mid_h,:mid_w], 'RGB')
    img_rt = Image.fromarray(img_np[:mid_h,mid_w:], 'RGB')
    img_lb = Image.fromarray(img_np[mid_h:,:mid_w], 'RGB')
    img_rb = Image.fromarray(img_np[mid_h:,mid_w:], 'RGB')
     
    img_lt.save("images_cut/"+img_name+"_1.tif")
    img_rt.save("images_cut/"+img_name+"_2.tif")
    img_lb.save("images_cut/"+img_name+"_3.tif")
    img_rb.save("images_cut/"+img_name+"_4.tif")
    
    
for f in os.listdir("images/"):
    img_name = f[:-4]
    cut_pics(img_name)





