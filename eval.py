import torch
import torch.nn.functional as F
import numpy as np
import pdb

from dice_loss import dice_coeff

def compute_iou(true, pred):
    true_mask = np.asanyarray(true.convert('L'), dtype = np.bool)
    pred_mask = np.asanyarray(pred.convert('L'), dtype = np.bool)
    union = np.sum(np.logical_or(true_mask, pred_mask))
    intersection = np.sum(np.logical_and(true_mask, pred_mask))
    iou = intersection/union
    return iou

def eval_net(net, dataset, gpu = False):
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]
        
        img = torch.from_numpy(img).unsqueeze(0)
#        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        
        if gpu:
            img = img.cuda()
#            true_mask = true_mask.cuda()
        
        mask_pred = net(img)[0]

       
       
        mask_pred = (mask_pred > 0.5).float()
        mask_pred_np = np.array(mask_pred)
        print(type(mask_pred_np),mask_pred_np.size,true_mask.shape)
        pdb.set_trace()
#        tot+=dice_coeff(input = mask_pred, target = true_mask).item()
    
    return tot/(i+1)