import sys,os
import torch
import numpy as np
from PIL import Image

# tensor1 = torch.tensor([[[2,4,1],[2,4,1]]])
# tensor2 = torch.tensor([[[2,1,1],[2,1,1]]]) 
# print(tensor1.shape,tensor2.shape) 
# print(tensor1[0].shape)  
img = Image.open("pig.png")
print(img.size)
img_np = np.array(img)
print(img_np.shape)
img_mid = Image.fromarray(img_np, 'RGB')
img_mid.save("pig_mid.png")
img_trans = np.transpose(img_np, [2,0,1])
img_output = Image.fromarray(img_trans, 'RGB')
img_output.save("pig_.png")
# print(tensor1.view(-1),tensor2.view(-1))
# print(torch.dot(tensor1.view(-1),tensor2.view(-1)))
# img = Image.open("big-pig-face.png")
# print(img.size)
# img1 = img.resize([300,300])
# img1.save("pig.png")
# img_np= np.array(img)
# print(img_np.shape)

# 
# img_crop = img.crop((0, 30, img.size[0], img.size[1]-30))
# print(img_crop.size)

# for f in os.listdir("images/"):
#     print(f)
    
# def get_ids(dir):
#     return (f[:-4] for f in os.listdir(dir))
# 
# def split_ids(ids, n=2):
#     return((id, i) for id in ids for i in range(n))
# 
# ids = get_ids("images/")
# ids = split_ids(ids)

