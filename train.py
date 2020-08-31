# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:47:05 2019

@author: liuliang
"""
# =============================================================================
# original package
# =============================================================================
import torch
import numpy as np
from torch import optim
from pathlib import Path
# =============================================================================
# creatived package
# =============================================================================
from model import LibraNet, weights_normal_init
from buffer import ReplayBuffer
from train_test import train_model,test_model

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# =============================================================================
# Parameters
# =============================================================================   
parameters = {'TRAIN_SKIP':100,
             'BUFFER_LENGTH':10000,
             'ERROR_RANGE':0.5,
             'GAMMA':0.9,
             'batch_size':128,
             'Interval_N':57,
             'step_log':0.1,
             'start_log':-2,
             'HV_NUMBER':8,
             'ACTION_NUMBER':9,
             'ERROR_SYSTEM':0,
             'means':[[108.25673428], [ 97.02240046], [ 93.37483706]]}
# =============================================================================
# Path setting
# =============================================================================
train_path ='data/Train/'
test_path ='data/Test/'      
# =============================================================================
# learning rate
# =============================================================================
learning_rate = 0.00001 * np.ones(100)
all_epoches = len(learning_rate)
# =============================================================================
# initialization 
# =============================================================================
minerror = np.zeros(2)
minerror[0] = 9999
minerror[1] = 9999

net = LibraNet(parameters) 
weights_normal_init(net, 0.01) 

if not Path("model_ckpt.pth.tar").is_file():
    epoch_last=0    
    print("Load pretrained model!")
    net.backbone.load_state_dict(torch.load('backbone.pth.tar')['state_dict'])
    print("Load finish!set pretrained paraments!")

else:
    print("Load check point model!")
    net.load_state_dict(torch.load('model_ckpt.pth.tar')['state_dict'])
    epoch_last = torch.load('model_ckpt.pth.tar')['epoch'] + 1
    minerror[0] = torch.load('model_ckpt.pth.tar')['mae']
    minerror[1] = torch.load('model_ckpt.pth.tar')['mse']
    
net = net.cuda()   
            
replay = ReplayBuffer(size=parameters['BUFFER_LENGTH'], vector_len_fv=512, vector_len_hv=parameters['HV_NUMBER'], batch_size=parameters['batch_size'])
             
for epoch in range(epoch_last, all_epoches): 
    net.DQN_faze.load_state_dict(net.DQN.state_dict())
    optimizer = optim.SGD([{'params':net.DQN.parameters(), 'lr':learning_rate[epoch]}])

    train_model(net, epoch, all_epoches, train_path, replay, optimizer, minerror, parameters)
    mae,mse = test_model(net, epoch, test_path, parameters)
    
    ##Save model
    if mae < minerror[0]:
        minerror[0] = mae
        minerror[1] = mse
            
        state_best = {
            'state_dict':net.state_dict(),
            'epoch':epoch,
            'mae':mae,
            'mse':mse
        }
        torch.save(state_best, 'model_best.pth.tar')
                
    state_ckpt = {
                'state_dict':net.state_dict(),
                'epoch':epoch,
                'mae':mae,
                'mse':mse
            }
    
    torch.save(state_ckpt, 'model_ckpt.pth.tar')
        
    print('mae=%.3f,mse=%.3f\n'%(mae, mse))
    
    f = open("result.txt", 'a') 
    f.write('EPOCH:%d, mae=%.4f,mse=%.4f\n'%(epoch, mae, mse))
    f.close()
    
print("Training finish!")
    
    