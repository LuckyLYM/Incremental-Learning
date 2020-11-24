import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import numpy as np
import quadprog
import miosqp
import scipy as sp
import scipy.sparse as spa
from .common import MLP, ResNet18
import random
import numpy as np


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
        self.n_inputs = n_inputs
        self.opt = optim.SGD(self.parameters(), args.lr)
        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda
        self.batch_size=args.batch_size
        self.n_iter = args.n_iter

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
        self.total_class=self.total_task*self.class_per_task
        self.class_nums  = [0 for i in range(self.total_class)]

        self.initialize_memory_buffer()

    def initialize_memory_buffer(self):
        self.sampled_class_data = []
        self.sampled_class_labs = []
        for i in range(self.total_class):
            class_data_buffer = torch.cuda.FloatTensor(self.n_sampled_memories, self.n_inputs)
            class_labs_buffer = torch.cuda.LongTensor(self.n_sampled_memories)    
            self.sampled_class_data.append(class_data_buffer)
            self.sampled_class_labs.append(class_labs_buffer)

    def forward(self, x, t=0):
        output = self.net(x)
        return output


    def update_reservoir(self,x,y):

        start_index=self.n_old_class
        end_index=self.n_class

        effective_new_data=None
        effective_new_labs=None

        for class_id in range(start_index,end_index):
            mask=torch.eq(y,class_id)
            class_data=x[mask]
            class_labs=y[mask]
            class_size=class_labs.size(0)            
            self.class_nums[class_id]+=class_size

            new_data,new_labs=self.update_reservoir_each_class(class_id,class_data,class_labs)
            if effective_new_labs is None:
                effective_new_data=new_data
                effective_new_labs=new_labs
            else:
                effective_new_data=torch.cat((effective_new_data,new_data),dim=0)
                effective_new_labs=torch.cat((effective_new_labs,new_labs),dim=0)

        return effective_new_data,effective_new_labs

    def update_reservoir_each_class(self,class_id,class_data,class_labs):

        class_size=class_labs.size(0)
        sampled_memory_data=self.sampled_class_data[class_id]
        sampled_memory_labs=self.sampled_class_labs[class_id]
        examples_seen_so_far=self.class_nums[class_id]
        effective_inds=[]

        for i in range(class_size):
            current_data=class_data[i]
            current_label=class_labs[i]
            if examples_seen_so_far<self.class_buffer_size:
                index=examples_seen_so_far
                sampled_memory_data[index]=current_data
                sampled_memory_labs[index]=current_label
                effective_inds.append(index)
            else:
                j = np.random.randint(0, examples_seen_so_far)
                if j<self.class_buffer_size:
                    sampled_memory_data[j]=current_data
                    sampled_memory_labs[j]=current_label
                    effective_inds.append(j)
            examples_seen_so_far+=1

        self.class_nums[class_id]=examples_seen_so_far
        effective_inds=list(dict.fromkeys(effective_inds))
        return sampled_memory_data[effective_inds],sampled_memory_labs[effective_inds]

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

        effective_new_data,effective_new_labs=self.update_reservoir(x,y)
        effective_size=effective_new_labs.size(0)
        print('effective size: %s'%effective_size)


        for iter_i in range(self.n_iter):

            if effective_size!=0:
                self.zero_grad()
                loss = self.ce(self.forward(effective_new_data), effective_new_labs)
                loss.backward()
                self.opt.step()
            
            if self.n_task==1 and self.class_nums[0]>=20:
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
