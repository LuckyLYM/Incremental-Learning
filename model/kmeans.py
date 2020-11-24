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
                
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.opt = optim.SGD(self.parameters(), args.lr)
        self.n_memories = args.n_memories                  # number of memories per task
        self.n_sampled_memories = args.n_sampled_memories  # number of sampled memories per task
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda
        self.batch_size=args.batch_size
        self.n_iter = args.n_iter
        self.pseudo=args.pseudo
        self.mode=args.mode


        # we allocate buffer for each class, not each task
        self.sampled_class_data = {}
        self.sampled_class_labs = {}
        self.sampled_class_features={}
        self.sampled_class_counter={}

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


    def shrink_buffer(self, class_id,sampled_memory_data, sampled_memory_labs, sampled_memory_counter,sampled_memory_features,pretrained):
        total_size=sampled_memory_counter.size(0)
        exceed_num=total_size-self.class_buffer_size
        old_features=sampled_memory_features[:self.class_buffer_size]
        new_features=sampled_memory_features[-exceed_num:]
        dist=self.dist_matrix(new_features,old_features)

        # idx0 is the index in the new feature list
        # idx1 is the index in the old feature list
        for idx0 in range(exceed_num):
            idx1=torch.argmin(dist[idx0])
            pt = sampled_memory_data[idx0+self.class_buffer_size]
            w = sampled_memory_counter[idx1]
            sampled_memory_data[idx1]=pt
            sampled_memory_counter[idx1]=w+1
            sampled_memory_features[idx1]=(new_features[idx0]+sampled_memory_features[idx1]*w)/(w+1)

        self.sampled_class_data[class_id]=sampled_memory_data[:self.class_buffer_size]
        self.sampled_class_labs[class_id]=sampled_memory_labs[:self.class_buffer_size]
        self.sampled_class_counter[class_id]=sampled_memory_counter[:self.class_buffer_size]
        self.sampled_class_features[class_id]=sampled_memory_features[:self.class_buffer_size]


    def update_kmeans_each_class(self,class_id,class_data,class_labs,class_features,pretrained,beta=2,alpha=1):
        sampled_size=class_labs.size(0)
        if class_id not in self.sampled_class_labs.keys():
            sampled_memory_data=class_data
            sampled_memory_labs=class_labs
            sampled_memory_features=class_features
            sampled_memory_counter = torch.ones(sampled_size).cuda() 

            self.sampled_class_data[class_id]=class_data
            self.sampled_class_labs[class_id]=class_labs
            self.sampled_class_features[class_id]=sampled_memory_features
            self.sampled_class_counter[class_id]=sampled_memory_counter

        else:
            sampled_memory_data=self.sampled_class_data[class_id]
            sampled_memory_labs=self.sampled_class_labs[class_id]
            sampled_memory_features=self.sampled_class_features[class_id]
            sampled_memory_counter=self.sampled_class_counter[class_id]

            sampled_memory_data=torch.cat((sampled_memory_data,class_data),dim=0)
            sampled_memory_labs=torch.cat((sampled_memory_labs,class_labs),dim=0)
            sampled_memory_features=torch.cat((sampled_memory_features,class_features),dim=0)
            sampled_memory_counter=torch.cat((sampled_memory_counter,torch.ones(sampled_size).cuda() ),dim=0)
            total_size=sampled_memory_labs.size(0)

            if total_size>self.class_buffer_size:
                self.shrink_buffer(class_id,sampled_memory_data,sampled_memory_labs,sampled_memory_counter,sampled_memory_features,pretrained)
            else:
                self.sampled_class_data[class_id]=sampled_memory_data
                self.sampled_class_labs[class_id]=sampled_memory_labs
                self.sampled_class_counter[class_id]=sampled_memory_counter
                self.sampled_class_features[class_id]=sampled_memory_features


    # Implement the online k-means algorithm introduced in
    # Memory Efficient Experience Replay for Streaming Learning. ICRA'19.
    def update_kmeans(self,pretrained,x,y):
        self.eval()
        start_index=self.n_old_class
        end_index=self.n_class
        if pretrained==None:
            new_features = self.net.feature_extractor(x).data
        else:
            new_features = pretrained.feature_extractor(x).data
        new_features = new_features/torch.norm(new_features, p=2,dim=1).unsqueeze(1)
        for class_id in range(start_index,end_index):
            mask=torch.eq(y,class_id)
            class_data=x[mask]
            class_labs=y[mask]
            class_features=new_features[mask]
            class_size=class_labs.size(0)            
            self.class_nums[class_id]+=class_size
            self.update_kmeans_each_class(class_id,class_data,class_labs,class_features,pretrained)

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
                # shrink the buffer size for each class
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
            
        self.update_kmeans(pretrained,x,y)