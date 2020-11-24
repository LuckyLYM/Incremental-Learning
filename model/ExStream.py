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

        # allocate ring buffer
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # we allocate buffer for each class, not each task
        self.sampled_class_data = {}
        self.sampled_class_labs = {}
        self.sampled_class_counter={}
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

        self.normalize = args.normalize


    def forward(self, x, t=0):
        output = self.net(x)
        return output
        
    def feature_and_output(self,x):
        return self.net.feature_and_output(x)

    # This implementation is copied from 
    # Memory Efficient Experience Replay for Streaming Learning. ICRA'19.
    def dist_matrix(self,H):

        with torch.no_grad():
            M, d = H.shape   # number of points, input shape
            H2 = torch.reshape(H, (M, 1, d))  # reshaping for broadcasting
            inside = H2 - H
            square_sub = torch.mul(inside, inside)  # square all elements
            psi = torch.sum(square_sub, dim=2)  # capacity x batch_size

            # infinity on diagonal
            mb = psi.shape[0]
            diag_vec = torch.ones(mb).cuda() * np.inf
            mask = torch.diag(torch.ones_like(diag_vec).cuda())
            psi = mask * torch.diag(diag_vec) + (1. - mask) * psi

        return psi


    # need to improve the efficiency here
    def shrink_buffer(self, class_id,sampled_memory_data, sampled_memory_labs, sampled_memory_counter,sampled_memory_features, pretrained):

        total_size=sampled_memory_counter.size(0)
        exceed_num=total_size-self.class_buffer_size
        class_features=sampled_memory_features

        for i in range(exceed_num):

            '''
                This is slow. But we can improve it.
                The first step is to test whether it is effective 
            '''

            dist = self.dist_matrix(class_features)
            idx  = torch.argmin(dist)
            idx0_= idx / total_size
            idx1_= idx % total_size
            idx0 = torch.min(idx0_,idx1_)
            idx1 = torch.max(idx0_,idx1_)
            dist[idx0][idx1]=dist[idx1][idx0]=np.inf

            pt1 = sampled_memory_data[idx0]
            pt2 = sampled_memory_data[idx1]
            w1 = sampled_memory_counter[idx0]
            w2 = sampled_memory_counter[idx1]


            merged_pt=pt1

            class_features[idx0]=(class_features[idx0]*w1+class_features[idx1]*w2)/(w1+w2)
            sampled_memory_data[idx0]=merged_pt
            sampled_memory_counter[idx0]=w1+w2

            # remove data point located at idx1
            inds=torch.tensor([i for i in range(total_size) if i!=idx1])
            sampled_memory_data=sampled_memory_data[inds]
            sampled_memory_counter=sampled_memory_counter[inds]
            class_features=class_features[inds]

            total_size-=1


        self.sampled_class_data[class_id]=sampled_memory_data
        self.sampled_class_labs[class_id]=sampled_memory_labs[:self.class_buffer_size]
        self.sampled_class_counter[class_id]=sampled_memory_counter
        self.sampled_class_features[class_id]=class_features

    def update_exstream_each_class(self,class_id,class_data,class_labs,class_features,pretrained,beta=2,alpha=1):
        
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
                #print('total size: %d  start shrinking buffer'%total_size)
                self.shrink_buffer(class_id,sampled_memory_data,sampled_memory_labs,sampled_memory_counter,sampled_memory_features,pretrained)
            else:
                # update the class buffer
                self.sampled_class_data[class_id]=sampled_memory_data
                self.sampled_class_labs[class_id]=sampled_memory_labs
                self.sampled_class_counter[class_id]=sampled_memory_counter
                self.sampled_class_features[class_id]=sampled_memory_features


    '''
        Implement the ExStream (Exemplar Streaming) method published in
        Memory Efficient Experience Replay for Streaming Learning. ICRA'19.
        This method will generate pseudo examples. seems very strange
        
        The implementation here is almost the same as the kmeans algorithn, except the
        shrink size function
    '''

    def update_exstream(self,pretrained):
        self.eval()
        start_index=self.n_old_class
        end_index=self.n_class

        if pretrained==None:
            new_features = self.net.feature_extractor(self.memory_data).data
        else:
            new_features = pretrained.feature_extractor(self.memory_data).data
        new_features = new_features/torch.norm(new_features, p=2,dim=1).unsqueeze(1)

        for class_id in range(start_index,end_index):
            mask=torch.eq(self.memory_labs,class_id)
            class_data=self.memory_data[mask]
            class_labs=self.memory_labs[mask]
            class_features=new_features[mask]
            class_size=class_labs.size(0)
            self.class_nums[class_id]+=class_size
            self.update_exstream_each_class(class_id,class_data,class_labs,class_features,pretrained)

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
            self.update_exstream(pretrained)