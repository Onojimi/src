#This is for training models, testing 1
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
from utils.load import get_ids, get_imgs_and_masks
from utils.utils import split_train_val, batch, normalize
import pdb



    
def train_net(net,
              epochs = 5,
              batch_size = 1,
              lr = 0.1,
              val_percent = 0.1,
              save_cp = False,
              gpu = True,
              img_scale = 0.5):
    
        img_dir = 'images_cut/'
        mask_dir = 'masks_cut/'
        checkpoint_dir ='checkpoints/'
        
        ids = get_ids(img_dir)      #['4488','6778','8767'...]
        
        iddataset = split_train_val(ids, val_percent)
        
        print('''
        Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))
        
        N_train = len(iddataset['train'])
        optimizer = optim.SGD(net.parameters(),
                              lr = lr,
                              momentum = 0.9,
                              weight_decay = 0.0005)
        
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            net.train()
        
            #reset the generators
            train = get_imgs_and_masks(iddataset['train'],img_dir,mask_dir, img_scale)     
            val = get_imgs_and_masks(iddataset['val'], img_dir, mask_dir, img_scale) 
            
            epoch_loss = 0
        
            for i, b in enumerate(batch(train, batch_size)):
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                true_masks = np.array([i[1] for i in b])
            
                imgs = torch.from_numpy(imgs)
                true_masks = torch.from_numpy(true_masks)
                
                true_masks = normalize(true_masks)
            
                if gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()
            
                masks_pred = net(imgs)
            
#                 print(true_masks.size())
#                 print(masks_pred.size())
#                 
#                 masks_pred_detach = masks_pred.cpu().detach().numpy()
#                 masks_pred_show = np.transpose(masks_pred_detach, [0,2,3,1])
#                 masks_pred_show = masks_pred_show*255
#                 
#                 print(masks_pred_show[0])
#                 mask_show1 = Image.fromarray(masks_pred_show[0].squeeze(), 'L')
#                 mask_show2 = Image.fromarray(masks_pred_show[1].squeeze(), 'L')
#                  
#                 mask_show1.show()
                
            
                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)
                loss = criterion(masks_probs_flat, true_masks_flat)
                epoch_loss += loss.item()
            
                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
        
            if 1:
                val_dice = eval_net(net, val, gpu)
                print('Validation Dice Coeff: {}'.format(val_dice))

            if save_cp:
                torch.save(net.state_dict(),
                           checkpoint_dir + 'CP{}.pth'.format(epoch + 1))
                print('Checkpoint {} saved !'.format(epoch + 1))       
  
def get_args():
    parser = OptionParser()
    parser.add_option('-e','--epochs',dest = 'epochs', default = 5, type = 'int',
                      help = 'number of epochs') 
    parser.add_option('-b','--batch-size', dest = 'batchsize', default = 5, type = 'int',
                      help = 'batchsize') 
    parser.add_option('-l', '--learning-rate', dest = 'lr', default = 0.1, type = float,
                      help = 'learning rate')
    parser.add_option('-g', '--gpu', dest = 'gpu', action = 'store_true', default = True, 
                      help = 'use cuda') 
    parser.add_option('-c', '--load', dest = 'load', default = False,
                      help = 'load file model')
    parser.add_option('-s', '--scale', dest = 'scale', default = 1, type = float,
                      help = 'downscaling factor of the images') 
    
    (options, args) = parser.parse_args() 
    return options  

if __name__ == '__main__':
    args = get_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = Unet(n_channels=3, n_classes=1)
    
#     net.cuda()
#     import pdb
#     from torchsummary import summary 
#     summary(net, (3,1000,1000))
#     pdb.set_trace()
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
        
    if args.gpu:
        if torch.cuda.device_count()>1:
            net = nn.DataParallel(net)
        net.cuda()
        
    try:
        train_net(net = net, 
                  epochs = args.epochs,
                  batch_size = args.batchsize, 
                  lr = args.lr, 
                  gpu = args.gpu, 
                  img_scale = args.scale)
        torch.save(net.state_dict(),'model_fin.pth')
        
    except KeyboardInterrupt:
        torch.save(net.state_dict(),'interrupt.pth')
        print('saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

