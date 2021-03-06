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

# no replay when training the first task
# keep the running estimate of mean features, and store 
# samples that closest to the running average

# Implement the Online K-means algorithm
# Just a simple try. I think it is not as effective as the ExStream approach.

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.is_cifar = ('cifar10' in args.data_file)
        if self.is_cifar:
            self.net = ResNet18(n_outputs, bias=args.bias)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
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

        # allocate ring buffer
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # we allocate buffer for each class, not each task
        self.sampled_class_data = {}
        self.sampled_class_labs = {}
        self.sampled_class_features={}
        self.sampled_class_counter={}
        self.gpu_id=args.gpu_id

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.n_task=0
        self.n_old_task=0
        self.sample_size_list=[]
        self.class_buffer_size=0
        self.n_class=0
        self.n_old_class=0
        self.total_task = args.total_task
        self.class_per_task=args.class_per_task
        self.class_nums  = [0 for i in range(self.total_task*self.class_per_task)]


    def forward(self, x, t=0):
        output = self.net(x)
        return output


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


    def shrink_buffer(self, class_id,sampled_memory_data, sampled_memory_labs, sampled_memory_counter,sampled_memory_features):
        
        total_size=sampled_memory_counter.size(0)
        exceed_num=total_size-self.class_buffer_size
        old_features=sampled_memory_features[:self.class_buffer_size]
        new_features=sampled_memory_features[-exceed_num:]
        dist=self.dist_matrix(new_features,old_features)
        effective_inds=[]

        # idx0 is the index in the new feature list
        # idx1 is the index in the old feature list
        for idx0 in range(exceed_num):
            idx1=torch.argmin(dist[idx0])
            pt = sampled_memory_data[idx0+self.class_buffer_size]
            w = sampled_memory_counter[idx1]

            sampled_memory_data[idx1]=pt
            sampled_memory_counter[idx1]=w+1
            sampled_memory_features[idx1]=(new_features[idx0]+sampled_memory_features[idx1]*w)/(w+1)
            effective_inds.append(idx1)

        self.sampled_class_data[class_id]=sampled_memory_data[:self.class_buffer_size]
        self.sampled_class_labs[class_id]=sampled_memory_labs[:self.class_buffer_size]
        self.sampled_class_counter[class_id]=sampled_memory_counter[:self.class_buffer_size]
        self.sampled_class_features[class_id]=sampled_memory_features[:self.class_buffer_size]
        return effective_inds

    def update_effective_data(self,effective_new_data,effective_new_labs,class_data,class_labs):
        
        if effective_new_data is None:
            effective_new_data=class_data
            effective_new_labs=class_labs
        else:
            effective_new_data=torch.cat((effective_new_data,class_data),dim=0)
            effective_new_labs=torch.cat((effective_new_labs,class_labs),dim=0)

        return effective_new_data,effective_new_labs


    def update_kmeans_each_class(self,class_id,class_data,class_labs,class_features,beta=2,alpha=1):
        
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
            return class_data,class_labs

        else:
            sampled_memory_data=self.sampled_class_data[class_id]
            sampled_memory_labs=self.sampled_class_labs[class_id]
            sampled_memory_features=self.sampled_class_features[class_id]
            sampled_memory_counter=self.sampled_class_counter[class_id]
            buffer_size=sampled_memory_labs.size(0)

            sampled_memory_data=torch.cat((sampled_memory_data,class_data),dim=0)
            sampled_memory_labs=torch.cat((sampled_memory_labs,class_labs),dim=0)
            sampled_memory_features=torch.cat((sampled_memory_features,class_features),dim=0)
            sampled_memory_counter=torch.cat((sampled_memory_counter,torch.ones(sampled_size).cuda() ),dim=0)
            total_size=sampled_memory_labs.size(0)

            if total_size>self.class_buffer_size:
                partial_effective_inds = list(range(buffer_size,self.class_buffer_size))
                effective_inds = self.shrink_buffer(class_id,sampled_memory_data,sampled_memory_labs,sampled_memory_counter,sampled_memory_features)
                
                
                effective_inds.extend(partial_effective_inds)
                effective_inds=list(dict.fromkeys(effective_inds))
                effective_new_data=self.sampled_class_data[class_id][effective_inds]
                effective_new_labs=self.sampled_class_labs[class_id][effective_inds]
                return effective_new_data, effective_new_labs

            else:
                self.sampled_class_data[class_id]=sampled_memory_data
                self.sampled_class_labs[class_id]=sampled_memory_labs
                self.sampled_class_counter[class_id]=sampled_memory_counter
                self.sampled_class_features[class_id]=sampled_memory_features

                return class_data,class_labs

    def update_kmeans(self,pretrained,x,y):
        self.eval()

        start_index=self.n_old_class
        end_index=self.n_class

        if pretrained==None:
            new_features = self.net.feature_extractor(x).data
        else:
            new_features = pretrained.feature_extractor(x).data

        effective_new_data= None
        effective_new_labs= None

        for class_id in range(start_index,end_index):
            mask=torch.eq(y,class_id)
            class_data=x[mask]
            class_labs=y[mask]
            class_features=new_features[mask]
            class_size=class_labs.size(0)
            self.class_nums[class_id]+=class_size
            
            new_data,new_labs=self.update_kmeans_each_class(class_id,class_data,class_labs,class_features)

            if effective_new_data is None:
                effective_new_data=new_data
                effective_new_labs=new_labs
            else:
                effective_new_data=torch.cat((effective_new_data,new_data),dim=0)
                effective_new_labs=torch.cat((effective_new_labs,new_labs),dim=0)

        return effective_new_data,effective_new_labs


    def observe(self, x, t, y, pretrained=None):
        
        if t!=self.old_task:
            self.old_task=t
            self.observed_tasks.append(t)
            self.n_old_task=self.n_task
            self.n_task+=1
            self.n_old_class=self.n_old_task*self.class_per_task
            self.n_class=self.n_task*self.class_per_task
            self.class_buffer_size=int(self.n_sampled_memories/self.n_class)


            if self.n_task>1:
                # determine the sample size from each class
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


        effective_new_data,effective_new_labs=self.update_kmeans(pretrained,x,y)
        effective_size=effective_new_labs.size(0)
        print('effective size: %d'%effective_size)


        self.train()

        for iter_i in range(self.n_iter):
            # update model on the new data
            self.zero_grad()
            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()
            

            if self.n_task==1 and self.class_nums[0]>=20:
       
                equal_sample_size=int(self.n_constraints/self.n_class)
                batch_x=None
                batch_y=None

                # sample from each new class
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

            #----update model on the old data----#
            elif self.n_task>1:

                batch_x=None
                batch_y=None

                # sample from each class
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
