import warnings
warnings.filterwarnings("ignore") 
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb
import numpy as np
import quadprog
import miosqp
import scipy as sp
import scipy.sparse as spa
from .common import MLP, ResNet18
import random
import numpy as np
import pretrained_cifar
import losses
from copy import deepcopy


# SDC with two parts regularization




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
        self.n_inputs = n_inputs
        self.opt = optim.SGD(self.parameters(), args.lr)
        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda
        self.batch_size=args.batch_size
        self.n_iter = args.n_iter
        self.alpha = args.alpha
        self.loss=args.loss
        self.feature_update=args.feature_update
        self.memory_update=args.memory_update
        self.initialize_criterion(args)


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



    def initialize_criterion(self,args):
        # loss_list=['triplet','triplet_no_hard_mining','center_triplet']
        if args.loss=='triplet_no_hard_mining' or args.loss=='triplet':
            self.criterion = losses.create(args.loss, margin=args.margin, num_instances=args.num_instances).cuda()
        elif args.loss=='center_triplet':
            self.criterion = losses.create(args.loss).cuda()           

    def forward(self, x, t=0):
        output = self.net(x)
        return output

    def feature_and_output(self,x):
        return self.net.feature_and_output(x)

    def freeze_model(self,model):
        for param in model.parameters():
            param.requires_grad = False
        return model


    def observe(self, x, t, y, pretrained=None):
        
        if t!=self.old_task:
            self.old_task=t
            self.observed_tasks.append(t)
            self.n_old_task=self.n_task
            self.n_task+=1
            self.n_old_class=self.n_old_task*self.class_per_task
            self.n_class=self.n_task*self.class_per_task

            if self.n_task>1:
                self.model_old = deepcopy(self.net)
                self.model_old.eval()
                self.model_old = self.freeze_model(self.model_old)



        for iter_i in range(self.n_iter):
            if self.n_task>1:
                self.zero_grad()
                feature,outputs=self.feature_and_output(x)
                loss1=F.cross_entropy(outputs, y)

                old_feature=self.model_old.feature_extractor(x).data
                if self.loss=='KL':
                    loss2=nn.KLDivLoss()(F.log_softmax(feature, dim=1),F.softmax(old_feature, dim=1))
                elif self.loss=='Euclidean':
                    loss2 = torch.sum((old_feature-feature).pow(2))/2.
                else:
                    loss2, _, _, _ = self.criterion(feature, y)
                loss = loss1*self.alpha + loss2*(1-self.alpha)
                loss.backward()
                self.opt.step()
            else:
                self.zero_grad()
                loss = self.ce(self.forward(x), y)
                loss.backward()
                self.opt.step()

