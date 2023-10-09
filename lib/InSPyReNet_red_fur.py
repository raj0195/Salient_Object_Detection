import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from lib.optim import *
from lib.modules.layers import *
from lib.modules.context_module import *
from lib.modules.attention_module import *
from lib.modules.decoder_module import *
from lib.InSPyReNet import *
from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinB

class InSPyReNet_change(nn.Module):
    def __init__(self, backbone,in_channels, depth=64, base_size=[384, 384], threshold=512, **kwargs):
        super(InSPyReNet_change, self).__init__()
        self.backbone = backbone
      #  self.model=pretrained_model
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        self.threshold = threshold
        
        #self.context1=pretrained_model.context1
        #self.context2=pretrained_model.context3
        #self.context3=pretrained_model.context3
        self.context1 = PAA_e(self.in_channels[0], self.depth, base_size=self.base_size, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, base_size=self.base_size, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, base_size=self.base_size, stage=2) ##
        #self.context4 = PAA_e(self.in_channels[3], self.depth, base_size=self.base_size, stage=3)
        #self.context5 = PAA_e(self.in_channels[4], self.depth, base_size=self.base_size, stage=4)

        self.decoder = PAA_d(self.depth, depth=self.depth, base_size=base_size, stage=1)
        #self.decoder=pretrained_model.decoder(in_

        self.attention0 = SICA(self.depth    , depth=self.depth, base_size=self.base_size, stage=0, lmap_in=True)
        self.attention1 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=1)
        #self.attention2 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=2              )
    
        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x, y, reduction='mean')
        self.pc_loss_fn  = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        
        self.image_pyramid = ImagePyramid(7, 1)
        
        self.transition0 = Transition(17)
        self.transition1 = Transition(9)
        self.transition2 = Transition(5)
        
        self.forward = self.forward_inference
        
    def to(self, device):
        self.image_pyramid.to(device)
        self.transition0.to(device)
        self.transition1.to(device)
        self.transition2.to(device)
        super(InSPyReNet_change, self).to(device)
        return self
    
    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            
        self.to(device="cuda:{}".format(idx))
        return self
    
    def train(self, mode=True):
        super(InSPyReNet_change, self).train(mode)
        self.forward = self.forward_train
        return self
    
    def eval(self):
        super(InSPyReNet_change, self).train(False)
        self.forward = self.forward_inference
        return self
    
    def forward_inspyre(self, x):
        B, _, H, W = x.shape
    
        x1, _, _, _, _ = self.backbone(x) ## only two outputs for different stages of the backbone
       
        x1 = self.context1(x1) #4
    
        x1 = self.res(x1, (H, W ))
        f2, d2 = self.decoder([x1]) #4
            
     
        
        out = dict()
        out['saliency'] = d2
        out['laplacian'] = []
        
        return out
    
    def forward_train(self, sample):
        x = sample['image']
        B, _, H, W = x.shape
        out = self.forward_inspyre(x)
        
        d2= out['saliency']
       # p1, p0    = out['laplacian']
        
        if type(sample) == dict and 'gt' in sample.keys() and sample['gt'] is not None:
            y = sample['gt']
                    #y1 = self.image_pyramid.reduce(y)
         
            
            loss =  self.sod_loss_fn(self.des(d2, (H, W)), self.des(y, (H, W)))
       
            
        else:
            loss = 0
            
        pred = torch.sigmoid(d2)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = loss
        sample['saliency'] = [d2]
        sample['laplacian'] = []
        return sample
    
    def forward_inference(self, sample):
        B, _, H, W = sample['image'].shape
        
        if self.threshold is None:
            #print('there')
            out = self.forward_inspyre(sample['image'])
            d2 = out['saliency']
            #p1, p0     = out['laplacian']
            
        elif (H <= self.threshold or W <= self.threshold):
            
            if 'image_resized' in sample.keys():
                out = self.forward_inspyre(sample['image_resized'])
            else:
                out = self.forward_inspyre(sample['image'])
            d3, d2, d1, d0 = out['saliency']
            p2, p1, p0     = out['laplacian']
        
        else:
            # LR Saliency Pyramid
            lr_out = self.forward_inspyre(sample['image_resized'])
            lr_d3, lr_d2, lr_d1, lr_d0 = lr_out['saliency']
            lr_p2, lr_p1, lr_p0      = lr_out['laplacian']
                
            # HR Saliency Pyramid
            hr_out = self.forward_inspyre(sample['image'])
            hr_d3, hr_d2, hr_d1, hr_d0 = hr_out['saliency']
            hr_p2, hr_p1, hr_p0      = hr_out['laplacian']
            
            # Pyramid Blending
            d3 = self.ret(lr_d0, hr_d3) 
            
            t2 = self.ret(self.transition2(d3), hr_p2)
            p2 = t2 * hr_p2
            d2 = self.image_pyramid.reconstruct(d3, p2)
            
            t1 = self.ret(self.transition1(d2), hr_p1)
            p1 = t1 * hr_p1
            d1 = self.image_pyramid.reconstruct(d2, p1)
            
            t0 = self.ret(self.transition0(d1), hr_p0)
            p0 = t0 * hr_p0
            d0 = self.image_pyramid.reconstruct(d1, p0)
            
        pred = torch.sigmoid(d2)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        sample['pred'] = pred
        sample['loss'] = 0
        sample['saliency'] = [d2]
        sample['laplacian'] = []
        return sample
    
def InSPyReNet_Res2Net50_change_fur(depth, pretrained, base_size, **kwargs):
    #opt=load_config('configs/InSPyReNet_Res2Net50.yaml')

#    model=eval(opt.Model.name)(**opt.Model)
 #   model_weights="/media/rtiwari/raj/inspyrenet_new/InSPyReNet/data/resnet_ever/latest.pth"
  #  model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')), strict=True)
  #  model.eval()
    return InSPyReNet_change(res2net50_v1b_26w_4s(pretrained=pretrained),[64, 256, 512, 1024, 2048], depth, base_size, **kwargs)

def InSPyReNet_SwinB(depth, pretrained, base_size, **kwargs):
    return InSPyReNet(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)