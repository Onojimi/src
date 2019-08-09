import os
import numpy as np
from PIL import Image

def cut_pics(img_name):
    img = Image.open("masks/"+img_name+"_mask.png")
    img_np = np.array(img)
    
    h = img_np.shape[0]
    w = img_np.shape[1]  
    
    mid_h = int(h/2)
    mid_w = int(w/2)
    
    img_lt = Image.fromarray(img_np[:mid_h,:mid_w], 'L')
    img_rt = Image.fromarray(img_np[:mid_h,mid_w:], 'L')
    img_lb = Image.fromarray(img_np[mid_h:,:mid_w], 'L')
    img_rb = Image.fromarray(img_np[mid_h:,mid_w:], 'L')
     
    img_lt.save("masks_cut/"+img_name+"_1_mask.png")
    img_rt.save("masks_cut/"+img_name+"_2_mask.png")
    img_lb.save("masks_cut/"+img_name+"_3_mask.png")
    img_rb.save("masks_cut/"+img_name+"_4_mask.png")
    
    
for f in os.listdir("masks/"):
    img_name = f[:4]
    cut_pics(img_name)





