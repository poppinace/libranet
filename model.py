# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:04:42 2018

@author: liuliang
"""
import torch.nn as nn
import torch
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint   
import numpy as np

class VGG16_BackBone(nn.Module):
    def __init__(self):
        super(VGG16_BackBone, self).__init__()
        self.layer0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)        
        self.layer1 = nn.ReLU(inplace=True)        
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.layer3 = nn.ReLU(inplace=True)            
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2) 
# =============================================================================        
        self.layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.layer6 = nn.ReLU(inplace=True)      
        self.layer7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  
        self.layer8 = nn.ReLU(inplace=True)              
        self.layer9 = nn.MaxPool2d(kernel_size=2, stride=2) 
# =============================================================================        
        self.layer10 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) 
        self.layer11 = nn.ReLU(inplace=True)      
        self.layer12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) 
        self.layer13 = nn.ReLU(inplace=True)      
        self.layer14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) 
        self.layer15 = nn.ReLU(inplace=True)              
        self.layer16 = nn.MaxPool2d(kernel_size=2, stride=2) 
               
# =============================================================================        
        self.layer17 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) 
        self.layer18 = nn.ReLU(inplace=True)      
        self.layer19 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer20 = nn.ReLU(inplace=True)      
        self.layer21 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer22 = nn.ReLU(inplace=True)              
        self.layer23 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
# =============================================================================
        self.layer24 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer25 = nn.ReLU(inplace=True)   
        self.layer26 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.layer27 = nn.ReLU(inplace=True)    
        self.layer28 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.layer29 = nn.ReLU(inplace=True)
        self.layer30 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)
                
    def forward(self, x):
        x = checkpoint(self.layer0,x)         
        x = self.layer1(x)  
        x = checkpoint(self.layer2,x)  
        x = self.layer3(x)  
        x = checkpoint(self.layer4,x)  
        x = checkpoint(self.layer5,x)  
        x = self.layer6(x)  
        x = checkpoint(self.layer7,x)  
        x = self.layer8(x)  
        x = checkpoint(self.layer9,x)  
        x = checkpoint(self.layer10,x)  
        x = self.layer11(x)  
        x = checkpoint(self.layer12,x)  
        x = self.layer13(x)  
        x = checkpoint(self.layer14,x)  
        x = self.layer15(x)  
        x = checkpoint(self.layer16,x)   
                 
        x = checkpoint(self.layer17,x)             
        x = self.layer18(x)  
        x = checkpoint(self.layer19,x) 
        x = self.layer20(x)  
        x = checkpoint(self.layer21,x)  
        x = self.layer22(x)          
        x = checkpoint(self.layer23,x)  
                         
        x = checkpoint(self.layer24,x)  
        x = self.layer25(x)  
        x = checkpoint(self.layer26,x)  
        x = self.layer27(x)  
        x = checkpoint(self.layer28,x)  
        x = self.layer29(x)    
        x = checkpoint(self.layer30,x)  
        return x

class DQN(nn.Module):
    def __init__(self, ACTION_NUMBER, HV_NUMBER):
        super(DQN, self).__init__()
        self.ACTION_NUMBER = ACTION_NUMBER
        self.layer1 = nn.Conv2d(in_channels=HV_NUMBER+512, out_channels=1024, kernel_size=1, padding=0)        
        self.layer2 = nn.ReLU(inplace=True)  
        
        self.layer3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding=0)        
        self.layer4 = nn.ReLU(inplace=True)  
                 
        self.layer5 = nn.Conv2d(in_channels=1024, out_channels=ACTION_NUMBER, kernel_size=1, padding=0)         
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.01)
                
    def forward(self, x, hv):                
        x = [x,hv]        
        x = torch.cat(x,1)
        del hv
        
        x = self.layer1(x) 
        x = self.layer2(x) 
        
        x = self.layer3(x) 
        x = self.layer4(x) 
                
        x = self.layer5(x) 
        return x
    
class LibraNet(nn.Module):
    def __init__(self, parameters):
        super(LibraNet, self).__init__()  
        #Weights definition
        Action1 = -10
        Action2 = -5
        Action3 = -2
        Action4 = -1
        Action5 = 1
        Action6 = 2
        Action7 = 5
        Action8 = 10
        Action9 = 999
        self.A = [Action1,Action2,Action3,Action4,Action5,Action6,Action7,Action8,Action9]
        self.A_mat = np.array(self.A)
        
        self.A_mat_h_w = np.expand_dims(np.expand_dims(self.A_mat, 1), 2)
        
        #Inverse discretization vector
        self.class2num = np.zeros(parameters['Interval_N'])
        for i in range(1, parameters['Interval_N']):
            if i == 1:
                lower = 0
            else:
                lower = np.exp((i - 2) * parameters['step_log'] + parameters['start_log'])
            upper = np.exp((i - 1) * parameters['step_log'] + parameters['start_log'])
            self.class2num[i] = (lower + upper) / 2
        
        #Network definition
        self.backbone = VGG16_BackBone()       
        self.DQN = DQN(parameters['ACTION_NUMBER'], parameters['HV_NUMBER'])
        self.DQN_faze = DQN(parameters['ACTION_NUMBER'], parameters['HV_NUMBER'])
        
    def get_feature( self, im_data=None):
        return self.backbone(im_data)
    
    def get_Q(self, feature=None, history_vectory=None):
        return self.DQN(feature,history_vectory) * 100
    
    def get_Q_faze(self, feature=None, history_vectory=None):
        return self.DQN_faze(feature, history_vectory) * 100
   
def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, dev)
