import torch
import torch.nn as nn
import numpy as np
import pdb
def compute_iou(true, pred):
    true_mask = np.asanyarray(true, dtype = np.bool)
    pred_mask = np.asanyarray(pred, dtype = np.bool)
    union = np.sum(np.logical_or(true_mask, pred_mask))
    intersection = np.sum(np.logical_and(true_mask, pred_mask))
    iou = intersection/union
    return iou

def eval_net(net, dataset, gpu = False):
    net.eval()
    iou = 0
    ls = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]
        true_mask /=255
        
        img = torch.from_numpy(img).unsqueeze(0)
#        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        
        if gpu:
            img = img.cuda()
#            true_mask = true_mask.cuda()
        
        mask_pred = net(img)[0]
       
        mask_pred = (mask_pred > 0.5).float()
        mask_pred_np = np.array(mask_pred.cpu()).squeeze(0)
        
        ls+= nn.BCELoss(mask_pred.view(-1),true_mask.view(-1)) 

        iou += compute_iou(true_mask,mask_pred_np)
    
    return iou/(i+1), ls/(i+1)