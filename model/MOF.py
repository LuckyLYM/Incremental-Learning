import sys
sys.path.append("../")
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
import pretrained_cifar
import utils



'''
    This is the adaptive data selector. This method combines reservoir sampling with MOF,
    and adaptive the selection strategy to the intermediate results.
    I am supposed to finish at least two strategies in this method. Let's move on and keep going.
'''


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens

        if 'mnist' in args.data_file:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
        else:
            if args.finetune=='None':
                self.net = ResNet18(n_outputs, bias=args.bias)
            else:
                self.net = pretrained_cifar.cifar_resnet20(pretrained=args.finetune,num_classes=n_outputs)
                
        self.distance_measure=args.distance
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.opt = optim.SGD(self.parameters(), args.lr)
        self.n_memories = args.n_memories                  # number of memories per task
        self.n_sampled_memories = args.n_sampled_memories  # number of sampled memories per task
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda
        self.batch_size=args.batch_size
        self.n_iter = args.n_iter
        self.mode=args.mode

        self.sampled_class_data = {}
        self.sampled_class_labs = {}
        self.sampled_class_features = {}

        # allocate counters
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

        # initialized for MOF
        self.mean_features = [0 for i in range(self.total_task*self.class_per_task)]
        self.class_nums  = [0 for i in range(self.total_task*self.class_per_task)]
        self.normalize = args.normalize


    def forward(self, x, t=0):
        output = self.net(x)
        return output

    def feature_and_output(self,x):
        return self.net.feature_and_output(x)

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


    def update_MOF(self,pretrained,x,y):
        self.eval()
        start_index=self.n_old_class
        end_index=self.n_class
        if pretrained==None:
            new_features = self.net.feature_extractor(x).data
        else:
            new_features = pretrained.feature_extractor(x).data
        new_features=new_features/torch.norm(new_features, p=2,dim=1).unsqueeze(1)

        for class_id in range(start_index,end_index):
            mask=torch.eq(y,class_id)
            class_data=x[mask]
            class_labs=y[mask]
            class_features=new_features[mask]
            class_size=class_labs.size(0)
            self.class_nums[class_id]+=class_size
            if class_id not in self.sampled_class_labs.keys():
                self.sampled_class_data[class_id]=class_data
                self.sampled_class_labs[class_id]=class_labs
                self.sampled_class_features[class_id]=class_features
                self.mean_features[class_id]=torch.mean(new_features[mask],dim=0,keepdim=True)                
            else:
                buffer_size=self.sampled_class_labs[class_id].size(0)
                total_size=class_size+buffer_size
                lack=self.class_buffer_size-buffer_size
                # bug version
                ratio=class_size/total_size
                # average version
                # ratio=class_size/self.class_nums[class_id]
                self.sampled_class_data[class_id]=torch.cat((self.sampled_class_data[class_id],class_data),dim=0)
                self.sampled_class_labs[class_id]=torch.cat((self.sampled_class_labs[class_id],class_labs),dim=0)
                self.sampled_class_features[class_id]=torch.cat((self.sampled_class_features[class_id],class_features),dim=0)
                self.mean_features[class_id]= self.mean_features[class_id]*(1-ratio)+torch.mean(new_features[mask],dim=0,keepdim=True)*ratio

                if total_size>self.class_buffer_size:
                    if self.distance_measure=='Euclidean':
                        dist=self.dist_matrix(self.sampled_class_features[class_id], self.mean_features[class_id]).squeeze()
                        sorted_inds=torch.argsort(dist,descending=False)
                    elif self.distance_measure=='cosine':
                        dist=(self.sampled_class_features[class_id]*self.mean_features[class_id]).sum(dim=1)
                        sorted_inds=torch.argsort(dist,descending=True)
                    else:
                        print('unsupported distance measure: ',self.distance_measure)
                    sorted_inds=sorted_inds[:self.class_buffer_size]
                    self.sampled_class_data[class_id]=self.sampled_class_data[class_id][sorted_inds]
                    self.sampled_class_labs[class_id]=self.sampled_class_labs[class_id][sorted_inds]
                    self.sampled_class_features[class_id]=self.sampled_class_features[class_id][sorted_inds]

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


    def single_batch_update(self,x,y):
        if self.n_task==1:
            self.zero_grad()
            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()
        else:
            batch_x=x
            batch_y=y
            for index in range(self.n_old_class):
                sample_size=self.sample_size_list[index]
                buffer_size=self.sampled_class_labs[index].size(0)
                random_inds=random.sample(range(0,buffer_size),sample_size)
                batch_x=torch.cat((batch_x,self.sampled_class_data[index][random_inds]),dim=0)
                batch_y=torch.cat((batch_y,self.sampled_class_labs[index][random_inds]),dim=0)
            self.zero_grad()
            loss = self.ce(self.forward(batch_x), batch_y)
            loss.backward()
            self.opt.step()

    def double_batch_update(self,x,y):
            self.zero_grad()
            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()

            if self.n_task==1 and self.class_nums[0]>=self.n_constraints:
                equal_sample_size=int(self.n_constraints/self.n_class)
                batch_x=None
                batch_y=None
                for class_id in range(self.n_class):
                    buffer_size=self.sampled_class_labs[class_id].size(0)
                    sample_size=min(buffer_size,equal_sample_size)
                    random_inds=random.sample(range(0,buffer_size),sample_size)
                    if batch_x is None:
                        batch_x=self.sampled_class_data[class_id][random_inds]
                        batch_y=self.sampled_class_labs[class_id][random_inds]
                    else:
                        batch_x=torch.cat((batch_x,self.sampled_class_data[class_id][random_inds]),dim=0)
                        batch_y=torch.cat((batch_y,self.sampled_class_labs[class_id][random_inds]),dim=0)
                self.zero_grad()
                loss = self.ce(self.forward(batch_x), batch_y)
                loss.backward()
                self.opt.step()

            elif self.n_task>1:
                batch_x=None
                batch_y=None
                for index in range(self.n_old_class):
                    sample_size=self.sample_size_list[index]
                    buffer_size=self.sampled_class_labs[index].size(0)
                    random_inds=random.sample(range(0,buffer_size),sample_size)
                    if batch_x is None:
                        batch_x=self.sampled_class_data[index][random_inds]
                        batch_y=self.sampled_class_labs[index][random_inds]
                    else:
                        batch_x=torch.cat((batch_x,self.sampled_class_data[index][random_inds]),dim=0)
                        batch_y=torch.cat((batch_y,self.sampled_class_labs[index][random_inds]),dim=0)
                self.zero_grad()
                loss = self.ce(self.forward(batch_x), batch_y)
                loss.backward()
                self.opt.step()

            

    def observe(self, x, t, y, pretrained=None):
        
        if t!=self.old_task:
            self.old_task=t
            self.observed_tasks.append(t)
            self.mem_cnt = 0
            self.n_old_task=self.n_task
            self.n_task+=1
            self.n_old_class=self.n_old_task*self.class_per_task
            self.n_class=self.n_task*self.class_per_task
            self.class_buffer_size=int(self.n_sampled_memories/self.n_class)

            if self.n_task>1:
                remainder=self.n_constraints%self.n_old_class
                quotient=int(self.n_constraints/self.n_old_class)
                self.sample_size_list=[quotient for _ in range(self.n_old_class)]
                for i in range(remainder):
                    self.sample_size_list[i]=self.sample_size_list[i]+1
                for index in range(self.n_old_class):
                    old_size=self.sampled_class_labs[index].size(0)
                    new_size=min(old_size,self.class_buffer_size)
                    self.sampled_class_data[index]=self.sampled_class_data[index][:new_size]
                    self.sampled_class_labs[index]=self.sampled_class_labs[index][:new_size]  


        for iter_i in range(self.n_iter):
            if self.mode=='single':
                self.single_batch_update(x,y)
            elif self.mode=='double':
                self.double_batch_update(x,y)

        self.update_MOF(pretrained,x,y)