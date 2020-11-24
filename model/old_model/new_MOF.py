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


        # we allocate buffer for each class, not each task
        self.sampled_class_data = {}
        self.sampled_class_labs = {}
        self.sampled_class_features = {}

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

        # initialized for MOF
        self.mean_features = [0 for i in range(self.total_task*self.class_per_task)]
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

    def update_effective_data(self,effective_new_data,effective_new_labs,class_data,class_labs):
        
        if effective_new_data is None:
            effective_new_data=class_data
            effective_new_labs=class_labs
        else:
            effective_new_data=torch.cat((effective_new_data,class_data),dim=0)
            effective_new_labs=torch.cat((effective_new_labs,class_labs),dim=0)

        return effective_new_data,effective_new_labs



    # we only select represetative points to update model
    def update_MOF(self,pretrained,x,y):

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

            # we assume that the new batch data size will not exceed the buffer size
            if class_id not in self.sampled_class_labs.keys():
                self.sampled_class_data[class_id]=class_data
                self.sampled_class_labs[class_id]=class_labs
                self.sampled_class_features[class_id]=class_features
                self.mean_features[class_id]=torch.mean(new_features[mask],dim=0,keepdim=True)    

                effective_new_data,effective_new_labs=self.update_effective_data(effective_new_data,effective_new_labs,class_data,class_labs)

            else:
                buffer_size=self.sampled_class_labs[class_id].size(0)
                total_size=class_size+buffer_size
                lack=self.class_buffer_size-buffer_size
                ratio=class_size/total_size
                
                self.sampled_class_data[class_id]=torch.cat((self.sampled_class_data[class_id],class_data),dim=0)
                self.sampled_class_labs[class_id]=torch.cat((self.sampled_class_labs[class_id],class_labs),dim=0)
                self.sampled_class_features[class_id]=torch.cat((self.sampled_class_features[class_id],class_features),dim=0)

                self.mean_features[class_id]= self.mean_features[class_id]*(1-ratio)+torch.mean(class_features,dim=0,keepdim=True)*ratio

                # print('updated mean features')
                # print(self.mean_features[class_id])
                # mean features is two dimensional

                if total_size>self.class_buffer_size:

                    dist=self.dist_matrix(self.sampled_class_features[class_id], self.mean_features[class_id]).squeeze()


                    sorted_inds=torch.argsort(dist,descending=False)[:self.class_buffer_size]
                    effective_inds=sorted_inds[sorted_inds>=buffer_size]
                    class_effective_new_data=self.sampled_class_data[class_id][effective_inds]
                    class_effective_new_labs=self.sampled_class_labs[class_id][effective_inds]

                    effective_new_data,effective_new_labs=self.update_effective_data(effective_new_data,effective_new_labs,class_effective_new_data,class_effective_new_labs)

                    self.sampled_class_data[class_id]=self.sampled_class_data[class_id][sorted_inds]
                    self.sampled_class_labs[class_id]=self.sampled_class_labs[class_id][sorted_inds]
                    self.sampled_class_features[class_id]=self.sampled_class_features[class_id][sorted_inds]

                else:
                    effective_new_data,effective_new_labs=self.update_effective_data(effective_new_data,effective_new_labs,class_data,class_labs)

        self.train()

        return effective_new_data, effective_new_labs

    def observe(self, x, t, y, pretrained=None):
        
        if t!=self.old_task:
            # update the counter and list
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



        effective_new_data,effective_new_labs=self.update_MOF(pretrained,x,y)
        effective_size=effective_new_labs.size(0)
        print('effective size: %d'%effective_size)


        for iter_i in range(self.n_iter):

            if effective_size!=0:
                self.zero_grad()
                loss = self.ce(self.forward(effective_new_data), effective_new_labs)
                loss.backward()
                self.opt.step()
            

            if self.n_task==1 and self.class_nums[0]>=30:
       
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