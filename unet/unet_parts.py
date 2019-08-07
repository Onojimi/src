# -*- coding: gb2312 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    #每一层下采样时都要做两次卷积，第一次channel增大第二次channel不变.
    #这是每一层中的基本操作
    def __init__(self,in_ch,out_ch):
        super(double_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, 3, padding =1),
            nn.BatchNorm2d(out_ch, 3, padding = 1),
            nn.ReLU(inplace = True),
            )
    
    def forward(self,x):
        x = self.conv(x)
        return x
    
class in_conv(nn.Module):
    '''把double_conv封装了一下,作为第一层内部的操作（无需上/下采样）'''
    def __init__(self,in_ch,out_ch):
        super(in_conv).__init__()
        self.conv = double_conv(in_ch,out_ch)
        
    def forward(self,x):
        x = self.conv(x)
        return x

class down(nn.Module):
    '''左侧的操作，先maxpool再double_conv。
        maxpool的核为2，就是每2×2取一个值，这样长和宽都变为原来的一半
        double_conv里，每一次的卷积核都为3×3，所以一次卷积完之后长/宽缩小了3-1个单位
    '''
    def __init__(self,in_ch,out_ch):
        super(down, in_ch, out_ch).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch,out_ch)
        )
    
    def forward(self,x):
        x = self.mpconv(x)
        return x
    
class up(nn.Module):
    '''先把图像的长和宽都扩大到原来的2倍（上采样/反卷积）
        r之后再用pad把图像周围扩大一圈（为下一步的卷积做准备）
        r再之后
    '''
    def __init__(self, in_ch, out_ch, bilinear = True):
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride = 2)
            
        self.conv = double_conv(in_ch,out_ch) 
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
           
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(outconv,self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,1)
        
    def forward(self,x):
        x = self.conv(x)
        return x