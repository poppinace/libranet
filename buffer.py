import numpy as np
import random

def sample_n_unique(sampling_f, n1):
    res = []
    while len(res) < n1:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, vector_len_fv,vector_len_hv,batch_size):
        
        self.size = size
        self.batch_size=batch_size
        self.next_idx      = 0
        self.num_in_buffer = 0
        self.state_fv  = np.zeros((size, vector_len_fv))
        self.state_hv  = np.zeros((size, vector_len_hv))
        self.action     = np.zeros((size,1))
        self.reward     = np.zeros((size,1))
        self.next_state_hv  = np.zeros((size, vector_len_hv))
        self.done       = np.zeros((size,1))
        self.flag_full =0

    def can_sample(self):
        
        return self.flag_full>0

    def out(self):
        
        assert self.can_sample()
        
        idxes = sample_n_unique(lambda: random.randint(0, 
                            self.size  - 2), self.batch_size)
        state_fv_batch  = self.state_fv[idxes]
        state_hv_batch  = self.state_hv[idxes]
        next_state_hv_batch  = self.next_state_hv[idxes]
        act_batch   = self.action[idxes]
        rew_batch   = self.reward[idxes]
        done_mask   = self.done[idxes]

        return state_fv_batch,state_hv_batch, act_batch, rew_batch,next_state_hv_batch, done_mask

    def put(self, state_fv,state_hv, action, reward,  next_state_hv,  done):
            
        length=len(state_fv)
        
        if self.size-self.num_in_buffer>length:
            
            self.state_fv[self.num_in_buffer:self.num_in_buffer+length,:]  = state_fv
            self.state_hv[self.num_in_buffer:self.num_in_buffer+length,:]  = state_hv
            self.action[self.num_in_buffer:self.num_in_buffer+length,:]  = action
            self.reward[self.num_in_buffer:self.num_in_buffer+length,:]  = reward
            self.next_state_hv[self.num_in_buffer:self.num_in_buffer+length,:]  = next_state_hv
            self.done[self.num_in_buffer:self.num_in_buffer+length,:]  = done
            
            self.num_in_buffer=self.num_in_buffer+length
            
        else:
            
            self.flag_full=1
            buffer_int=self.size-self.num_in_buffer
            self.state_fv[self.num_in_buffer:self.size,:]  = state_fv[0:buffer_int,:]
            self.state_hv[self.num_in_buffer:self.size,:]  = state_hv[0:buffer_int,:]
            self.action[self.num_in_buffer:self.size,:]  = action[0:buffer_int,:]
            self.reward[self.num_in_buffer:self.size,:]  = reward[0:buffer_int,:]
            self.next_state_hv[self.num_in_buffer:self.size,:]  = next_state_hv[0:buffer_int,:]
            self.done[self.num_in_buffer:self.size,:]  = done[0:buffer_int,:]
            
            buffer_int2=length-buffer_int
            self.state_fv[0:buffer_int2,:]  = state_fv[buffer_int:length,:]
            self.state_hv[0:buffer_int2,:]  = state_hv[buffer_int:length,:]
            self.action[0:buffer_int2,:]  = action[buffer_int:length,:]
            self.reward[0:buffer_int2,:]  = reward[buffer_int:length,:]
            self.next_state_hv[0:buffer_int2,:]  = next_state_hv[buffer_int:length,:]
            self.done[0:buffer_int2,:]  = done[buffer_int:length,:]
            
            self.num_in_buffer =buffer_int2