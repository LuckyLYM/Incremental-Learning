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

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = ('cifar10' in args.data_file)
        m = miosqp.MIOSQP()
        self.solver = m
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

        # allocate ring buffer
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        self.added_index = self.n_sampled_memories
        # allocate buffer for the current task
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        # allocate buffer for each task
        self.sampled_task_data = {}
        self.sampled_task_labs = {}
        # allocate selected constraints
        self.constraints_data = None
        self.constraints_labs = None
        self.cluster_distance = 0
        # old grads to measure changes
        self.old_mem_grads = None
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()
        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.n_task=0
        self.n_old_task=0
        self.sample_size_list=[]
        self.task_buffer_size=0


    def forward(self, x, t=0):
        output = self.net(x)
        return output


    def select_k_centers(self,pretrained,beta=2,alpha=1):
        self.eval()
        if self.sampled_memory_data is None:
            self.sampled_memory_data = self.memory_data[0].unsqueeze(0).clone()
            self.sampled_memory_labs = self.memory_labs[0].unsqueeze(0).clone()
            new_memories_data=self.memory_data[1:].clone()
            new_memories_labs = self.memory_labs[1:].clone()
        else:
            new_memories_data=self.memory_data.clone()
            new_memories_labs = self.memory_labs.clone()

        # sampled_memory_data store the old data
        # new_memories_data store the new coming data

        #--------improvement needed here---------#
        # a small change here we can improve the efficiency here

        #---------output probability--------#
        #new_mem_features = self.net(new_memories_data).data
        #samples_mem_features = self.net(self.sampled_memory_data).data

        if pretrained==None:
            #---------self penultimate layer features--------#
            new_mem_features = self.net.feature_extractor(new_memories_data).data
            samples_mem_features = self.net.feature_extractor(self.sampled_memory_data).data
        else:
            #------pretrained penultimate layer features-----#
            new_mem_features = pretrained.feature_extractor(new_memories_data).data
            samples_mem_features = pretrained.feature_extractor(self.sampled_memory_data).data          

        new_dist=self.dist_matrix(new_mem_features, samples_mem_features)

        #intra_distance: rhe initial value is 0
        if self.cluster_distance==0:
            intra_dist=self.dist_matrix(samples_mem_features)
            max_dis=torch.max(intra_dist)
            eye=(torch.eye(intra_dist.size(0))*max_dis)
            if self.gpu:
                eye=eye.cuda()
            self.cluster_distance=alpha*torch.min(intra_dist+eye)#

        added_indes=[]
        for new_mem_index in range(new_mem_features.size(0)):
            if torch.min(new_dist[new_mem_index])>self.cluster_distance:
                added_indes.append(new_mem_index)
        #print("length of added inds",len(added_indes))

        # shrink the buffer size
        if (len(added_indes)+self.sampled_memory_data.size(0))>self.task_buffer_size:
            init_points=torch.cat((self.sampled_memory_data,new_memories_data[added_indes]),dim=0)
            init_points_labels=torch.cat((self.sampled_memory_labs,new_memories_labs[added_indes]),dim=0)
            init_points_feat=torch.cat((samples_mem_features,new_mem_features[added_indes]),dim=0)
            est_mem_size=init_points_feat.size(0)
            init_feat_dist=self.dist_matrix(init_points_feat)
            eye=torch.eye(init_feat_dist.size(0))
            if self.gpu:
                eye=eye.cuda()
            self.cluster_distance = torch.min(init_feat_dist+eye*torch.max(init_feat_dist))
            while est_mem_size>self.task_buffer_size:
                self.cluster_distance=self.cluster_distance*beta
                first_ind=torch.randint(0,init_points_feat.size(0),(1,))[0]
                cent_inds=[first_ind.item()]
                for feat_indx in range(init_points_feat.size(0)) :
                    if torch.min(init_feat_dist[feat_indx][cent_inds])>self.cluster_distance:
                        cent_inds.append(feat_indx)
                est_mem_size=len(cent_inds)
            #print("BUFFER SIZE,",est_mem_size)
            self.sampled_memory_data=init_points[cent_inds]
            self.sampled_memory_labs = init_points_labels[cent_inds]
        else:
            self.sampled_memory_data=torch.cat((self.sampled_memory_data,new_memories_data[added_indes]),dim=0)
            self.sampled_memory_labs=torch.cat((self.sampled_memory_labs,new_memories_labs[added_indes]),dim=0)

        # update the task buffer
        task_id=self.observed_tasks[self.n_task-1]
        self.sampled_task_data[task_id]=self.sampled_memory_data
        self.sampled_task_labs[task_id]=self.sampled_memory_labs

    def print_task_buffer(self):
        for task_id in self.sampled_task_data:
            print("task_id: %d buffer_size: %d"%(task_id,len(self.sampled_task_data[task_id])))

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

    #---problem setting---#
    # task incremental learning, task id is consecutive increasing
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
                    old_size=self.sampled_task_labs[task_id].size(0)
                    new_size=min(old_size,self.task_buffer_size)
                    random_inds=random.sample(range(0,old_size),new_size)
                    self.sampled_task_data[task_id]=self.sampled_task_data[task_id][random_inds]
                    self.sampled_task_labs[task_id]=self.sampled_task_labs[task_id][random_inds]  

            # intialize the cluster for new task
            self.sampled_memory_data=None
            self.sampled_memory_labs=None
            self.cluster_distance=0

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
                    buffer_size=self.sampled_task_labs[task_id].size(0)

                    # I need to fix the k-center algorithm here. it is so stupid
                    random_inds=random.sample(range(0,buffer_size),sample_size)

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
            self.select_k_centers(pretrained)