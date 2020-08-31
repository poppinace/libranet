# -*- coding: utf-8 -*-
import torch
import numpy as np
import time
import cv2
import pandas as pd
import math 
import scipy.io as sio
import os
from torchvision import transforms

def train_model(net, epoch, all_epoches, train_path, replay, optimizer, minerror, parameters):
    
    train_img = train_path + 'image/'
    train_gt = train_path + 'gt_classmap/'
    train_dir = os.listdir(train_img)      
    train_number = len(train_dir)
  
    EPSOLON = max(0.1, 1 - epoch * 0.05)
    net.eval()
    loss_train = 0
    number_deal = 0
        
    speed_c_image = 0
    speed_number_image = 1
    
    start_image = time.time()
    number_rest = 0
    
    toTensor = transforms.ToTensor()   
    means = torch.FloatTensor( np.array(parameters['means']) / 255) .unsqueeze_(0).unsqueeze_(2).cuda()
    for image_index in range(0, 1):
        print_T=0
        if image_index>0:
            end_image = time.time()
            speed_image = 1 / (end_image - start_image)
            speed_c_image += speed_image
            speed_number_image += 1
        
        start_image = time.time()
       
        if image_index==0 or epoch==-1 :
            print('[Epoch {:.2f}] {:.2f} / {:.2f}'.format(epoch, image_index, train_number))
                 
        image_name = train_img + str(image_index+1) +'.jpg'
        img = cv2.imread(image_name) 
        dot_name = train_gt + str(image_index+1) + '.csv' 
        
        featuremap_t = []
        featuremap_save = []
            
        den = np.array(pd.read_csv(dot_name, sep=',', header=None))        

        h = int(img.shape[0]/32)
        w = int(img.shape[1]/32)
                      
        mask_last = np.zeros((h, w))
        mask_last = mask_last.astype(np.int8)
        count_rem = np.zeros((h, w))  
        hv_save = np.zeros((h, w, parameters['HV_NUMBER']))
        
        img = toTensor(img).unsqueeze(0).float().cuda() - means
                
        featuremap_t = net.get_feature(im_data=img)
        featuremap_save = featuremap_t[0].data.cpu().numpy()
        featuremap_save = np.swapaxes( np.swapaxes(featuremap_save, 0, 2), 0, 1)
        
        for step_hv in range(0, parameters['HV_NUMBER']):                 
            reward_map = np.zeros((h, w)) 
            
            net.eval()     
                                       
            hv = torch.from_numpy( hv_save.transpose((2, 0, 1)) ).unsqueeze(0).float().cuda()
                            
            old_Q = net.get_Q(feature=featuremap_t, history_vectory=hv)
        
            old_qval = old_Q[0].data.cpu().numpy()                                       
            
            error_last = abs(den - count_rem)  
            q_t = -old_qval
            sort = q_t.argsort(axis=0)
            
            start_ind_random = -1 * np.ones((h, w))
            end_ind_random = -1 * np.ones((h, w))
            
            mask_max_find = np.zeros((h, w))
            action_max = np.zeros((h, w))
                        
            ##Exploration
            for recycle_ind in range(0, parameters['ACTION_NUMBER']):
                ##########################random##############################################
                if recycle_ind < parameters['ACTION_NUMBER'] - 1:
                    start_mask_random = ( (count_rem + net.A[recycle_ind] >= 0) & (start_ind_random == -1) )
                    start_ind_random[start_mask_random] = recycle_ind
    
                    end_mask_random = ( count_rem + net.A[recycle_ind] < parameters['Interval_N'] )
                    end_ind_random[end_mask_random] = recycle_ind                
                    
                maskselect_end = (sort[recycle_ind]==parameters['ACTION_NUMBER']-1)
                action_sort = sort[recycle_ind]
                
                A_sort = np.squeeze(net.A_mat[action_sort])
                
                _ind_max = (( (count_rem + A_sort < parameters['Interval_N']) & (count_rem + A_sort >= 0) | maskselect_end) & (mask_max_find==0) ) & (mask_last==0)
                action_max[_ind_max] = action_max[_ind_max] + sort[recycle_ind] [_ind_max]
                mask_max_find = mask_max_find + ( (count_rem + A_sort < parameters['Interval_N']) & (count_rem + A_sort >= 0) | maskselect_end ).astype(np.int8)
        
            action_random = (start_ind_random + (end_ind_random + 2 - start_ind_random ) * np.random.rand(h, w)).astype(np.int8)
            
            random_select =  (np.random.rand(h, w) < EPSOLON).astype(np.int8)
            action_fusion = random_select * action_random + (1-random_select) * action_max
            ######################################reward############################################
            optimal_action = np.zeros((h, w))
            
            count_after_every_action = np.expand_dims(count_rem, 0) + net.A_mat_h_w[0:parameters['ACTION_NUMBER']-1, :, :]
            error_every_action = abs(np.expand_dims(den, 0) - count_after_every_action)
            optimal_action_mid = error_every_action.argsort(axis=0)
            optimal_action = optimal_action_mid[0,:,:]
            
            optimal_action[error_last<=parameters['ERROR_SYSTEM']] = parameters['ACTION_NUMBER'] - 1
            mask_select_end = (action_fusion == parameters['ACTION_NUMBER'] - 1).astype(np.int8)
            mask_now = mask_last.copy()
            mask_now = mask_now | mask_select_end
            
            count_rem = count_rem + (1 - mask_select_end) * (1 - mask_last) * (np.squeeze(net.A_mat_h_w[action_fusion.astype(np.int8)]))
            
            error_now = abs(den - count_rem)  
            hv_next = hv_save.copy()
            hv_next[:,:,step_hv] = action_fusion + 1            
                        
            ##Reward computation
            if step_hv != parameters['HV_NUMBER'] - 1:
                mask_in_range = (count_rem <= den * (1 + parameters['ERROR_RANGE'])).astype(np.int8)
                mask_error_decrease = (error_last > error_now).astype(np.int8)
                mask_optimal = (action_fusion == optimal_action).astype(np.int8)
                mask_could_end_last = (error_last <= parameters['ERROR_SYSTEM']).astype(np.int8)
                
                ##ending reward
                reward_map = mask_select_end * mask_could_end_last * 5 + mask_select_end * (1 - mask_could_end_last) * -5
                ##guiding reward
                reward_map = reward_map + (1 - mask_select_end) * mask_in_range * mask_error_decrease * mask_optimal * 3
                reward_map = reward_map + (1 - mask_select_end) * mask_in_range * mask_error_decrease * (1 - mask_optimal) * 1
                reward_map = reward_map + (1 - mask_select_end) * mask_in_range * (1 - mask_error_decrease) * -1
                ##squeeze guiding reward
                reward_map = reward_map + (1 - mask_select_end) * (1 - mask_in_range) * mask_error_decrease * mask_optimal * -1
                reward_map = reward_map + (1 - mask_select_end) * (1 - mask_in_range) * mask_error_decrease * (1 - mask_optimal) * -3
                reward_map = reward_map + (1 - mask_select_end) * (1 - mask_in_range) * (1 - mask_error_decrease) * -3
            else:
                mask_select_end = np.ones((h, w))                
                mask_could_end_now = (error_now <= parameters['ERROR_SYSTEM']).astype(np.int8)
                reward_map = mask_could_end_now * 5 + (1 - mask_could_end_now) * -5
            
            ##hard sample mining
            mask_drop = ((np.random.rand(h, w) < 0.5).astype(np.int8)) * ((error_last <= 1).astype(np.int8))
                
            if ((1-mask_last)*(1-mask_drop)).sum()<=1:
                continue
            
            state_fv = featuremap_save.reshape((featuremap_save.shape[0] * featuremap_save.shape[1], featuremap_save.shape[2]))
            state_hv = hv_save.reshape((hv_save.shape[0] * hv_save.shape[1], hv_save.shape[2]))
            action = action_fusion.reshape((action_fusion.shape[0] * action_fusion.shape[1], 1))
            reward = reward_map.reshape((reward_map.shape[0] * reward_map.shape[1], 1))
            next_state_hv = hv_next.reshape((hv_next.shape[0] * hv_next.shape[1], hv_next.shape[2]))
            done = mask_select_end.reshape((mask_select_end.shape[0] * mask_select_end.shape[1], 1))
            mask_last_batch = mask_last.reshape((mask_last.shape[0] * mask_last.shape[1], 1))
            mask_drop =  mask_drop.reshape((mask_drop.shape[0] * mask_drop.shape[1], 1))
                        
            state_fv = state_fv[np.squeeze(mask_last_batch == 0)]
            state_hv = state_hv[np.squeeze(mask_last_batch == 0)]
            action = action[np.squeeze(mask_last_batch == 0)]
            reward = reward[np.squeeze(mask_last_batch == 0)]
            next_state_hv = next_state_hv[np.squeeze(mask_last_batch == 0)]
            done = done[np.squeeze(mask_last_batch == 0)]
            mask_drop = mask_drop[np.squeeze(mask_last_batch == 0)]
            
            state_fv = state_fv[np.squeeze(mask_drop == 0)]
            state_hv = state_hv[np.squeeze(mask_drop == 0)]
            action = action[np.squeeze(mask_drop == 0)]
            reward = reward[np.squeeze(mask_drop == 0)]
            next_state_hv = next_state_hv[np.squeeze(mask_drop == 0)]
            done = done[np.squeeze(mask_drop == 0)]
            
            ##send to buffer
            if not replay.can_sample():
                #if buffer is not full
                replay.put(state_fv, state_hv, action, reward, next_state_hv, done)
            else:                
                #if buffer is full
                number_this_batch = len(state_fv)
                point_start = 0
                point_end = 0
                rest = number_this_batch + number_rest
                while rest>0:  
                    #train when every 100 samples are sent to buffer 
                    if rest < parameters['TRAIN_SKIP']: 
                        replay.put(state_fv[point_start:number_this_batch,:],\
                                   state_hv[point_start:number_this_batch,:],\
                                   action[point_start:number_this_batch],\
                                   reward[point_start:number_this_batch],\
                                   next_state_hv[point_start:number_this_batch,:],\
                                   done[point_start:number_this_batch])                        
                        number_rest=rest
                        rest=0
                    else:
                        point_end = min(point_end + parameters['TRAIN_SKIP'] - number_rest, number_this_batch)
                        number_rest = 0
                        
                        replay.put(state_fv[point_start:point_end,:],\
                                   state_hv[point_start:point_end,:],\
                                   action[point_start:point_end],\
                                   reward[point_start:point_end],\
                                   next_state_hv[point_start:point_end,:],\
                                   done[point_start:point_end])
                        point_start = point_end
                        rest = number_this_batch-point_end
                        
                        net.train()     
                        state_fv_batch, state_hv_batch, act_batch, rew_batch, next_state_hv_batch, done_mask = replay.out()
                                     
                        state_fv_batch = torch.FloatTensor(state_fv_batch).cuda().unsqueeze(2).unsqueeze(3)
                        state_hv_batch = torch.FloatTensor(state_hv_batch).cuda().unsqueeze(2).unsqueeze(3)
                        act_batch = torch.LongTensor(act_batch).cuda().unsqueeze(2).unsqueeze(3)
                        rew_batch = torch.FloatTensor(rew_batch).cuda().unsqueeze(2).unsqueeze(3)
                        next_state_hv_batch = torch.FloatTensor(next_state_hv_batch).cuda().unsqueeze(2).unsqueeze(3)
                        done_mask = torch.FloatTensor(done_mask).cuda().unsqueeze(2).unsqueeze(3)
                        
                        newQ = net.get_Q_faze(feature=state_fv_batch, history_vectory=next_state_hv_batch)
                        newQ =  newQ.data.max(1)[0].unsqueeze(1)
                        target_Q = newQ * parameters['GAMMA'] * (1 - done_mask) + rew_batch
                                    
                        eval_Q = net.get_Q(feature=state_fv_batch, history_vectory=state_hv_batch)
                        eval_Q = eval_Q.gather(1,act_batch)
                        
                        loss = (eval_Q - target_Q.detach()).abs().mean()
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()  
                        
                        loss_train += loss.item()
                            
                        number_deal = number_deal+1
                                                    
                        net.eval()     
                        
                        if (image_index%10==1 and print_T==0) or epoch==-1:  
                            print_T=1
                            print('Epoch:{}/{},image:{}/{},speed:{:.2f},Mae:{:.2f},Mse:{:.2f}, loss:{:.3f}'.format(
                               int(epoch),
                               int(all_epoches),
                               int(image_index),
                               int(train_number),
                               speed_c_image/speed_number_image,                                    
                               minerror[0],
                               minerror[1],
                               loss_train/number_deal
                               ))         
                 
            hv_save =  hv_next.copy()
            mask_last = mask_now.copy()   
            if (1-mask_now).sum()==0:
                break                
                    
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

    
