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
import sys

# enable replay when training the first class
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
        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda
        self.batch_size=args.batch_size
        self.n_iter = args.n_iter

        # allocate ring buffer
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        # allocate buffer for the current task
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        # allocate buffer for each task
        self.sampled_task_data = {}
        self.sampled_task_labs = {}
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()
        

        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.n_task=0
        self.n_old_task=0
        self.task_buffer_size=0
        self.sample_size_list=[]


    def forward(self, x, t=0):
        output = self.net(x)
        return output

    def select_random_samples(self):

        if self.sampled_memory_data is None:
            self.sampled_memory_data = self.memory_data.clone()
            self.sampled_memory_labs = self.memory_labs.clone()

        else:
            sampled_memory_size=self.sampled_memory_labs.size(0)
            total_size=sampled_memory_size+self.n_memories

            self.sampled_memory_data = torch.cat(
                (self.sampled_memory_data, self.memory_data), dim=0)
            self.sampled_memory_labs = torch.cat(
                (self.sampled_memory_labs, self.memory_labs), dim=0)

            if total_size>self.task_buffer_size:
                shuffeled_inds=torch.randperm(total_size)
                self.sampled_memory_data = self.sampled_memory_data[shuffeled_inds[0:self.task_buffer_size]]
                self.sampled_memory_labs = self.sampled_memory_labs[shuffeled_inds[0:self.task_buffer_size]]

        # update the task buffer
        task_id=self.observed_tasks[self.n_task-1]
        self.sampled_task_data[task_id]=self.sampled_memory_data
        self.sampled_task_labs[task_id]=self.sampled_memory_labs


    def observe(self, x, t, y, pretrained=None):

        # identify a new task
        if t!=self.old_task:
            # update the counter and list
            self.old_task=t
            self.observed_tasks.append(t)
            self.mem_cnt = 0
            self.n_old_task=self.n_task
            self.n_task+=1
            self.task_buffer_size=int(self.n_sampled_memories/self.n_task)

            if self.n_task>1:
                # determine the sample size from each task
                remainder=self.n_constraints%self.n_old_task
                quotient=int(self.n_constraints/self.n_old_task)
                self.sample_size_list=[quotient for _ in range(self.n_old_task)]
                for i in range(remainder):
                    self.sample_size_list[i]=self.sample_size_list[i]+1

                # shrink the buffer size
                for index in range(self.n_old_task):
                    task_id=self.observed_tasks[index]
                    self.sampled_task_data[task_id]=self.sampled_task_data[task_id][:self.task_buffer_size]
                    self.sampled_task_labs[task_id]=self.sampled_task_labs[task_id][:self.task_buffer_size]

            # intialize the cluster for new task
            self.sampled_memory_data=None
            self.sampled_memory_labs=None

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz


        for iter_i in range(self.n_iter):
            # update model on the new data
            self.zero_grad()
            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()
            


            if self.n_task==1 and self.sampled_memory_data is not None:
       
                size=self.sampled_memory_labs.size(0)
                random_inds=random.sample(range(0,size),self.n_constraints)
                task_id=self.observed_tasks[0]
                
                '''
                print("task_id: %d size： %d "%(task_id,size))
                print(random_inds)
                print(self.sampled_task_data[task_id][:size])
                print(self.sampled_task_labs[task_id][:size])
                '''

                batch_x=self.sampled_task_data[task_id][random_inds]
                batch_y=self.sampled_task_labs[task_id][random_inds]
                self.zero_grad()
                loss = self.ce(self.forward(batch_x), batch_y)
                loss.backward()
                self.opt.step()

            #----update model on the old data----#
            elif self.n_task>1:

                batch_x=None
                batch_y=None

                # sample from each task
                for index in range(self.n_old_task):
                    task_id=self.observed_tasks[index]
                    sample_size=self.sample_size_list[index]
                    random_inds=random.sample(range(0,self.task_buffer_size),sample_size)

                    if batch_x is None:
                        batch_x=self.sampled_task_data[task_id][random_inds]
                        batch_y=self.sampled_task_labs[task_id][random_inds]
                    else:
                        batch_x=torch.cat((batch_x,self.sampled_task_data[task_id][random_inds]),dim=0)
                        batch_y=torch.cat((batch_y,self.sampled_task_labs[task_id][random_inds]),dim=0)

                
                self.zero_grad()
                loss = self.ce(self.forward(batch_x), batch_y)
                loss.backward()
                self.opt.step()



            
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
            self.select_random_samples()
