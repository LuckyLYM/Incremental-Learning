import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import random
import numpy as np
import quadprog
import miosqp
import scipy as sp
import scipy.sparse as spa
from .common import MLP, ResNet18

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()

        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.n_task=0
        self.n_old_task=0
        self.sample_size_list=[]
        self.class_buffer_size=0
        self.n_class=0
        self.n_old_class=0
        self.total_task = args.total_task
        self.class_per_task=args.class_per_task
        self.feature_dim=args.feature_dim

        self.mean_features = torch.zeros(self.total_task*self.class_per_task,self.feature_dim)
        self.mean_features=self.mean_features.cuda()
        self.class_nums  = [0 for i in range(self.total_task*self.class_per_task)]
        self.normalize = args.normalize

    def dist_matrix(self,x,y=None):
        if y is None:
            y=x.clone()
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, 2).sum(2)
        return dist


    def update_MOF(self,x,y,pretrained):
        start_index=self.n_old_class
        end_index=self.n_class

        new_features = pretrained.feature_extractor(x).data
        new_features = new_features/torch.norm(new_features, p=2,dim=1).unsqueeze(1)

        for class_id in range(start_index,end_index):
            mask=torch.eq(y,class_id)
            class_features=new_features[mask]
            class_size=class_features.size(0)
            if self.class_nums[class_id]==0:
                self.mean_features[class_id]=torch.mean(new_features[mask],dim=0,keepdim=True)                
            else:
                ratio=class_size/(class_size+self.class_nums[class_id])
                self.mean_features[class_id]= self.mean_features[class_id]*(1-ratio)+torch.mean(new_features[mask],dim=0,keepdim=True)*ratio
            self.class_nums[class_id]+=class_size

            #print('class_id %d'%class_id)
            #print(self.mean_features[class_id])



    # we use the feature mean to do classifcation instead of model training.
    def observe(self, x, t, y, pretrained=None):
        if t!=self.old_task:
            self.old_task=t
            self.observed_tasks.append(t)
            self.n_old_task=self.n_task
            self.n_task+=1
            self.n_old_class=self.n_old_task*self.class_per_task
            self.n_class=self.n_task*self.class_per_task
        self.update_MOF(x,y,pretrained)