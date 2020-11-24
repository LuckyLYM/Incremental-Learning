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

        # initialized for k-center
        self.class_cluster_distance  = [0 for i in range(self.total_task*self.class_per_task)]


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

    def select_k_centers_each_class(self,class_id,class_data,class_labs,class_features,beta=2,alpha=1):
        
        if class_id not in self.sampled_class_labs.keys():
            sampled_memory_data=class_data[0].unsqueeze(0)
            sampled_memory_labs=class_labs[0].unsqueeze(0)

            new_memories_data=class_data[1:]
            new_memories_labs=class_labs[1:]

            new_mem_features=class_features[1:]
            sampled_memory_features=class_features[0].unsqueeze(0) 
        else:
            new_memories_data=class_data
            new_memories_labs=class_labs
            new_mem_features=class_features

            sampled_memory_data=self.sampled_class_data[class_id]
            sampled_memory_labs=self.sampled_class_labs[class_id]
            sampled_memory_features=self.sampled_class_features[class_id]


        new_dist=self.dist_matrix(new_mem_features, sampled_memory_features)
        cluster_distance=self.class_cluster_distance[class_id]

        if cluster_distance==0:
            intra_dist=self.dist_matrix(sampled_memory_features)
            max_dis=torch.max(intra_dist)
            eye=(torch.eye(intra_dist.size(0))*max_dis)
            if self.gpu:
                eye=eye.cuda()
            cluster_distance=alpha*torch.min(intra_dist+eye)#

        added_indes=[]
        for new_mem_index in range(new_mem_features.size(0)):
            if torch.min(new_dist[new_mem_index])>cluster_distance:
                added_indes.append(new_mem_index)

        if (len(added_indes)+sampled_memory_data.size(0))>self.class_buffer_size:
            init_points=torch.cat((sampled_memory_data,new_memories_data[added_indes]),dim=0)
            init_points_labels=torch.cat((sampled_memory_labs,new_memories_labs[added_indes]),dim=0)
            init_points_feat=torch.cat((sampled_memory_features,new_mem_features[added_indes]),dim=0)

            est_mem_size=init_points_feat.size(0)
            init_feat_dist=self.dist_matrix(init_points_feat)
            eye=torch.eye(init_feat_dist.size(0))
            if self.gpu:
                eye=eye.cuda()
            cluster_distance = torch.min(init_feat_dist+eye*torch.max(init_feat_dist))
            while est_mem_size>self.class_buffer_size:
                cluster_distance=cluster_distance*beta
                first_ind=torch.randint(0,init_points_feat.size(0),(1,))[0]
                cent_inds=[first_ind.item()]
                for feat_indx in range(init_points_feat.size(0)) :
                    if torch.min(init_feat_dist[feat_indx][cent_inds])>cluster_distance:
                        cent_inds.append(feat_indx)
                est_mem_size=len(cent_inds)

            #print("Class ID: %d  BUFFER SIZE: %d"%(class_id,est_mem_size))
            sampled_memory_data=init_points[cent_inds]
            sampled_memory_labs = init_points_labels[cent_inds]
            sampled_memory_features= init_points_feat[cent_inds]
        else:
            sampled_memory_data=torch.cat((sampled_memory_data,new_memories_data[added_indes]),dim=0)
            sampled_memory_labs=torch.cat((sampled_memory_labs,new_memories_labs[added_indes]),dim=0)
            sampled_memory_features=torch.cat((sampled_memory_features,new_mem_features[added_indes]),dim=0)
            #print("Class ID: %d  BUFFER SIZE: %d"%(class_id,sampled_memory_labs.size(0)))

        # update the class buffer
        self.sampled_class_data[class_id]=sampled_memory_data
        self.sampled_class_labs[class_id]=sampled_memory_labs
        self.sampled_class_features[class_id]=sampled_memory_features
        self.class_cluster_distance[class_id]=cluster_distance


    # I modify the method shown in the paper: 
    # Continual Learning with Tiny Episodic Memories. ICML'19.
    def select_k_centers(self,pretrained):
        self.eval()

        # get class index
        start_index=self.n_old_class
        end_index=self.n_class

        # get features of input new data
        if pretrained==None:
            new_features = self.net.feature_extractor(self.memory_data).data
        else:
            new_features = pretrained.feature_extractor(self.memory_data).data

        # split the samples for each class
        for class_id in range(start_index,end_index):
            mask=torch.eq(self.memory_labs,class_id)
            class_data=self.memory_data[mask]
            class_labs=self.memory_labs[mask]
            class_features=new_features[mask]

            class_size=class_labs.size(0)
            self.class_nums[class_id]+=class_size

            self.select_k_centers_each_class(class_id,class_data,class_labs,class_features)



    #dist[i, j] = ||x[i,:] - y[j,:]||^2
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





    def observe(self, x, t, y, pretrained=None):
        
        if t!=self.old_task:
            # update the counter and list
            self.old_task=t
            self.observed_tasks.append(t)
            self.mem_cnt = 0
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

        # update the ring buffer
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

            
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
            self.select_k_centers(pretrained)