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
        self.opt = optim.SGD(self.parameters(), args.lr)
        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda
        self.batch_size=args.batch_size
        self.n_iter = args.n_iter


        self.sampled_memory_data = torch.FloatTensor(self.n_sampled_memories, n_inputs)
        self.sampled_memory_labs = torch.LongTensor(self.n_sampled_memories)
        if args.cuda:
            self.sampled_memory_data = self.sampled_memory_data.cuda()
            self.sampled_memory_labs = self.sampled_memory_labs.cuda()
        self.mem_cnt = 0
        self.examples_seen_so_far=0

    def forward(self, x, t=0):
        output = self.net(x)
        return output

    # ring buffer version
    def update_reservoir(self,x,y):

        size=y.size(0)

        for i in range(size):
            current_data=x[i]
            current_label=y[i]
            if self.examples_seen_so_far<self.n_sampled_memories:
                index=self.examples_seen_so_far
                self.sampled_memory_data[index]=current_data
                self.sampled_memory_labs[index]=current_label
            else:
                j = np.random.randint(0, self.examples_seen_so_far)
                if j<self.n_sampled_memories:
                    self.sampled_memory_data[j]=current_data
                    self.sampled_memory_labs[j]=current_label

            self.examples_seen_so_far+=1

    # we use real ring buffer scheme instead
    def observe(self, x, t, y, pretrained=None):
 
        for iter_i in range(self.n_iter):
            # update model on the new data
            self.zero_grad()
            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()
            

            if self.examples_seen_so_far> self.n_constraints:

                size=min(self.examples_seen_so_far,self.n_sampled_memories)
                random_batch_inds=random.sample(range(0,size),self.n_constraints)
                batch_x=self.sampled_memory_data[random_batch_inds]
                batch_y = self.sampled_memory_labs[random_batch_inds]
                
                self.zero_grad()
                loss = self.ce(self.forward(batch_x), batch_y)
                loss.backward()
                self.opt.step()


        self.update_reservoir(x,y)

