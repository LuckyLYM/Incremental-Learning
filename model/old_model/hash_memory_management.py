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

        if 'mnist' in args.data_file:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
        else:
            self.net = ResNet18(n_outputs, bias=args.bias)

        self.n_inputs=n_inputs
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
        self.memory_data = torch.FloatTensor(self.n_memories, self.n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.ring_cnt = 0
        self.n_task=0
        self.n_old_task=0
        self.sample_size_list=[]
        self.class_buffer_size=0
        self.n_class=0
        self.n_old_class=0
        self.total_task = args.total_task
        self.class_per_task=args.class_per_task
        self.total_class=self.total_task*self.class_per_task
        self.mem_cnt=[0 for i in range(self.total_class)]
        self.class_nums=[0 for i in range(self.total_class)]


        # defined for hash clustering
        self.projected_dim=args.projected_dim
        self.n_hash_table= args.n_hash_table
        self.hashs=[]
        self.class_buckets=[]
        self.buckets=[]    # use this as a reference
        self.feature_dim=args.feature_dim


        self.initialize_memory_buffer()
        self.initialize_hash()

    def initialize_memory_buffer(self):

        # we allocate buffer for each class
        self.sampled_class_data = []
        self.sampled_class_labs = []
        self.sampled_class_features= []
        self.sampled_class_counter= []

        for i in range(self.total_class):

            class_data_buffer = torch.cuda.FloatTensor(self.n_sampled_memories, self.n_inputs)
            class_labs_buffer = torch.cuda.LongTensor(self.n_sampled_memories)
            class_counter_buffer = torch.cuda.ShortTensor(self.n_sampled_memories)
            class_feature_buffer =torch.cuda.FloatTensor(self.n_sampled_memories, self.feature_dim)
            
            self.sampled_class_data.append(class_data_buffer)
            self.sampled_class_labs.append(class_labs_buffer)
            self.sampled_class_features.append(class_feature_buffer)
            self.sampled_class_counter.append(class_counter_buffer)



    def initialize_hash(self):
        self.get_projection_matrix()
        self.get_hash_buckets()

    # We need a set of hash tables for each class
    def get_projection_matrix(self):
        for i in range(self.n_hash_table):
            projection=torch.randn(self.feature_dim,self.projected_dim).cuda()       
            self.hashs.append(projection)

    def get_hash_buckets(self):
        for i in range(self.total_class):
            self.class_buckets.append([])
            for _ in range(self.n_hash_table):
                hash_table=dict()
                self.class_buckets[i].append(hash_table)

        print('number of hash tables: %d'%self.n_hash_table)
        print('number of projected dimension: %d'%self.projected_dim)


    def get_hash_vector(self,x,hash_id):

        size=x.size(0)
        projection=self.hashs[hash_id]
        values = x.mm(projection)
        return [''.join(['1' if x > 0.0 else '0' for x in values[i,:]]) for i in range(size)]


    def delete_hash_vector(self,x,data_index):
        for i in range(self.n_hash_table):
            bucket_keys=self.get_hash_vector(x,i)
            for key in bucket_keys:
                self.buckets[i][key].remove(data_index)
                #print('remove hash index, hash table id: %d key: %s index: %d'%(i,key,data_index))

    def update_hash(self,new_x,old_x,index):

        self.delete_hash_vector(old_x,index)
        self.store_hash_vector(new_x,index)


    def store_hash_vector(self,x,data_index):

        start_index=data_index
        for i in range(self.n_hash_table):
            bucket_keys=self.get_hash_vector(x,i)
            for key in bucket_keys:
                if not key in self.buckets[i]:
                    self.buckets[i][key]=[]
                self.buckets[i][key].append(data_index)
                #print('add hash index, hash table id: %d key: %s index: %d start_index: %d'%(i,key,data_index,start_index))

                data_index+=1

            data_index=start_index


    def get_bucket_content(self,hash_id,bucket_key):
        if bucket_key in self.buckets[hash_id]:
            return self.buckets[hash_id]
        else:
            return []

    def get_candidates(self,v):
        '''
            input: v is an un-hashed vector
            return idex of candidate points in the memory buffer
        '''
        candidates=set()
        for hash_id in range(self.n_hash_table):
            bucket_key=self.get_hash_vector(v,hash_id)[0]
            bucket_content=self.get_bucket_content(hash_id,bucket_key)
            candidates.update(bucket_content)

        return list(candidates)

    def get_nearest_neighbor(self,x,neighbors):
        '''
            return the index of the nearest neighbor
        '''
        candidates=self.get_candidates(x)
        neighbor_size=len(candidates)
        #print('candidate size: %d'%neighbor_size)


        if neighbor_size==0:
            index= random.randint(0,self.class_buffer_size-1)
            print('no neighbor found, return random index %d'%index)
            return index

        elif neighbor_size==1:
            return candidates[0]

        else:
            dist=torch.pow(neighbors-x,2).sum(1)
            index=torch.argmin(dist)
            return index

    def forward(self, x, t=0):
        output = self.net(x)
        return output


    def shrink_buffer(self, class_id,new_data, new_labs,new_features):
        
        new_batch_size=new_labs.size(0)
        old_features=self.sampled_class_features[class_id]
        old_data=self.sampled_class_data[class_id]
        old_counter=self.sampled_class_counter[class_id]

        for idx0 in range(new_batch_size):
            new_feature=new_features[idx0].unsqueeze(0)
            idx1=self.get_nearest_neighbor(new_feature,old_features)
            idx1=int(idx1)

            pt = new_data[idx0]
            w = old_counter[idx1]
            old_feature=old_features[idx1]
            average_feature=(new_features[idx0]+old_feature*w)/(1+w)
            self.update_hash(average_feature.unsqueeze(0),old_feature.unsqueeze(0),idx1)

            old_data[idx1].copy_(pt)
            old_counter[idx1]=w+1
            old_features[idx1].copy_(average_feature)


    def update_kmeans_each_class(self,class_id,class_data,class_labs,class_features):
        
        sample_size=class_labs.size(0)
        buffer_cnt=self.mem_cnt[class_id]
        concat_size=sample_size+buffer_cnt

        if concat_size<=self.class_buffer_size:

            self.sampled_class_data[class_id][buffer_cnt:concat_size].copy_(class_data)
            self.sampled_class_labs[class_id][buffer_cnt:concat_size].copy_(class_labs)
            self.sampled_class_features[class_id][buffer_cnt:concat_size].copy_(class_features)
            self.sampled_class_counter[class_id][buffer_cnt:concat_size]=1 
            self.mem_cnt[class_id]=concat_size
            self.store_hash_vector(class_features,buffer_cnt)

        else:
            if buffer_cnt<self.class_buffer_size:
                residual=self.class_buffer_size-buffer_cnt
                self.sampled_class_data[class_id][buffer_cnt:self.class_buffer_size].copy_(class_data[:residual])
                self.sampled_class_labs[class_id][buffer_cnt:self.class_buffer_size].copy_(class_labs[:residual])
                self.sampled_class_features[class_id][buffer_cnt:self.class_buffer_size].copy_(class_features[:residual])
                self.sampled_class_counter[class_id][buffer_cnt:self.class_buffer_size]=1 
                self.mem_cnt[class_id]=self.class_buffer_size
                self.store_hash_vector(class_features[:residual],buffer_cnt)        

                class_data=class_data[residual:]
                class_labs=class_labs[residual:]
                class_features=class_features[residual:]

            self.shrink_buffer(class_id,class_data,class_labs,class_features)


    # Implement the online k-means algorithm introduced in
    # Memory Efficient Experience Replay for Streaming Learning. ICRA'19.
    def update_kmeans(self,pretrained):
        self.eval()

        start_index=self.n_old_class
        end_index=self.n_class

        if pretrained==None:
            new_features = self.net.feature_extractor(self.memory_data).data
        else:
            new_features = pretrained.feature_extractor(self.memory_data).data

        for class_id in range(start_index,end_index):
            self.buckets=self.class_buckets[class_id]
            mask=torch.eq(self.memory_labs,class_id)
            class_data=self.memory_data[mask]
            class_labs=self.memory_labs[mask]
            class_features=new_features[mask]
            class_size=class_labs.size(0)
            self.class_nums[class_id]+=class_size

            self.update_kmeans_each_class(class_id,class_data,class_labs,class_features)




    def observe(self, x, t, y, pretrained=None):
        
        if t!=self.old_task:
            self.old_task=t
            self.observed_tasks.append(t)
            self.ring_cnt = 0
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
                    old_size=self.mem_cnt[index]
                    new_size=min(old_size, self.class_buffer_size)
                    self.mem_cnt[index]=new_size
                    self.sampled_class_data[index]=self.sampled_class_data[index][:new_size]
                    self.sampled_class_labs[index]=self.sampled_class_labs[index][:new_size]  


        bsz = y.data.size(0)
        endcnt = min(self.ring_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.ring_cnt
        self.memory_data[self.ring_cnt: endcnt].copy_(x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[self.ring_cnt] = y.data[0]
        else:
            self.memory_labs[self.ring_cnt: endcnt].copy_(y.data[: effbsz])
        self.ring_cnt += effbsz


        for iter_i in range(self.n_iter):
            self.zero_grad()
            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()

            if self.n_task==1 and self.class_nums[0]>=20:
       
                equal_sample_size=int(self.n_constraints/self.n_class)
                batch_x=None
                batch_y=None

                for class_id in range(self.n_class):
                    buffer_size=self.mem_cnt[class_id]
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
                
                for class_id in range(self.n_old_class):
                    sample_size=self.sample_size_list[class_id]
                    buffer_size=self.mem_cnt[class_id]
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

            
        if self.ring_cnt == self.n_memories:
            self.ring_cnt = 0
            self.update_kmeans(pretrained)