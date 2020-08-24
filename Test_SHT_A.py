# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:47:05 2019

@author: liuliang
"""
# =============================================================================
# original package
# =============================================================================
import torch
# =============================================================================
# creatived package
# =============================================================================
from model import LibraNet
from train_test import test_model
       
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

test_path ='data/Test/'  
epoch=0

net = LibraNet(parameters) 
net.load_state_dict(torch.load('trained_model/LibraNet_SHT_A.pth.tar')['state_dict'])
net.cuda()

print('Test SHT PART A Start!')
mae,mse = test_model(net, epoch, test_path, parameters)
print('mae=%.3f,mse=%.3f\n'%(mae, mse))
print('Test SHT PART A Finish!')
