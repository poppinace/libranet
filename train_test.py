# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch
import numpy as np
import time
import cv2
import pandas as pd
import math 
import scipy.io as sio
import os
from torchvision import transforms
                    
def test_model(net, epoch, test_path, parameters):
    
    test_img = test_path + 'images/'
    test_gt = test_path + 'ground_truth/'
    
    files_test_im = os.listdir(test_img)      
    data_test_number = len(files_test_im)
    
    count_save = np.zeros((data_test_number,2))
    net.eval() 
    
    toTensor = transforms.ToTensor()   
    means = torch.FloatTensor(np.array(parameters['means']) / 255).unsqueeze(0).unsqueeze(2).cuda()
    
    for i in range(0,data_test_number):   
        gt_path = test_gt + 'GT_IMG_'+ str(i+1)+'.mat'           
        gt = sio.loadmat(gt_path)
                    
        img_name = test_img+ 'IMG_'+ str(i+1)+'.jpg'           
        Img = cv2.imread(img_name)
        
        h = Img.shape[0]
        w = Img.shape[1]
                           
        gt = len(gt['image_info'][0][0][0][0][0])
        
        ht = int(32*int(h/32))    
        wt = int(32*int(w/32))
        if ht != h:
            ht = int(32 * (int(h / 32) + 1))  
        if wt != w:
            wt = int(32 * (int(w / 32) + 1))  
            
        ho = int(ht/32)
        wo = int(wt/32)
                                                        
        Img_t = np.zeros((ht, wt,3))
        Img_t[0:h, 0:w, :] = Img.copy()
        Img = Img_t.astype(np.uint8)
                
        img = toTensor(Img).unsqueeze(0).cuda()-means
                                                 
        featuremap_t = []        
        class_rem = np.zeros((ho, wo))          
        hv_save = np.zeros((ho, wo, parameters['HV_NUMBER']))
                
        mask_last = np.zeros((ho, wo))
        mask_last = mask_last.astype(np.int8)
        
        featuremap_t = net.get_feature(im_data=img)
        for step in range(0, parameters['HV_NUMBER']): 
            
            hv = torch.from_numpy(hv_save.transpose((2, 0, 1))).unsqueeze_(0).float().cuda()
            
            Q = net.get_Q(feature=featuremap_t, history_vectory=hv)
                        
            Q = -Q[0].data.cpu().numpy()  
            sort = Q.argsort(axis=0)
            
            action_max = np.zeros((ho, wo))
            
            mask_max_find = np.zeros((ho,wo))
            for recycle_ind in range(0,parameters['ACTION_NUMBER']):
                maskselect_end = (sort[recycle_ind] == parameters['ACTION_NUMBER']-1)
                action_sort = sort[recycle_ind]
                    
                A_sort = np.squeeze(net.A_mat[action_sort])
                
                _ind_max = (((class_rem + A_sort <  parameters['Interval_N']) & (class_rem +A_sort >= 0) | maskselect_end) & ( mask_max_find == 0)) & (mask_last == 0)
                action_max[_ind_max] = action_max[_ind_max] + sort[recycle_ind] [_ind_max]
                mask_max_find = mask_max_find + ((class_rem + A_sort <  parameters['Interval_N']) & (class_rem +A_sort >= 0) | maskselect_end).astype(np.int8)
            
            mask_select_end=(action_max == parameters['ACTION_NUMBER']-1).astype(np.int8)
            class_rem = class_rem + (1 - mask_select_end) * (1 - mask_last) * (np.squeeze(net.A_mat_h_w[action_max.astype(np.int8)]))
            
            hv_save[:, :, step] = action_max+1 
            mask_now = mask_last.copy()
            mask_now = mask_now | mask_select_end
            mask_last = mask_now.copy()
            if (1 - mask_last).sum() == 0:
                break         
        
        count_rem = net.class2num[class_rem.astype(np.int8)]
        est = count_rem.sum()
        print('Testing {}/{}, GT:{}, EST:{}'.format(i, data_test_number, gt, int(est*100)/100))
        count_save[i,0] = gt
        count_save[i,1] = est 
                    
    w0 = count_save[:,0]
    w1 = count_save[:,1]
    
    mae = np.mean(abs(w0 - w1))   
    mse = math.sqrt(sum((w0 - w1) * (w0 - w1)) / data_test_number)
    return mae, mse

    