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
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()
        self.mem_cnt = 0

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

            if total_size>self.n_sampled_memories:
                shuffeled_inds=torch.randperm(total_size)
                self.sampled_memory_data = self.sampled_memory_data[shuffeled_inds[0:self.n_sampled_memories]]
                self.sampled_memory_labs = self.sampled_memory_labs[shuffeled_inds[0:self.n_sampled_memories]]


    def observe(self, x, t, y, pretrained=None):
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
            
            # update the model on random samples from sampled_memory_data
            if self.sampled_memory_data is not None:
                size=self.sampled_memory_labs.size(0)
                random_batch_inds=random.sample(range(0,size),self.n_constraints)
                batch_x=self.sampled_memory_data[random_batch_inds]
                batch_y = self.sampled_memory_labs[random_batch_inds]
                
                self.zero_grad()
                loss = self.ce(self.forward(batch_x), batch_y)
                loss.backward()
                self.opt.step()


        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
            self.select_random_samples()