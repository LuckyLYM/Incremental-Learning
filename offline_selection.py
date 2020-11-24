import sys 
sys.path.append("..") 
import pickle
import datetime
import argparse
import random
import time
import os
import numpy as np
import torch
import numpy as np
from kmeans_pytorch import kmeans
from model import pretrained_cifar

def dist_matrix(x,y=None):
    if y is None:
        y=x.clone()
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return dist

def get_features(class_data,batch_size,sample_size,pretrained):
    class_size=class_data.size(0)
    class_features=None
    class_cnt=0
    n_batch=int(class_size/batch_size)
    for i in range(n_batch):
        batch_data=class_data[class_cnt:class_cnt+batch_size]
        batch_data.cuda()
        batch_features = pretrained.feature_extractor(batch_data).data
        batch_features = batch_features/torch.norm(batch_features, p=2,dim=1).unsqueeze(1)
        if class_features is None:
            class_features=batch_features
        else:
            class_features=torch.cat((class_features,batch_features),dim=0)
        class_cnt+=batch_size
    return class_features

def update_MOF(class_data,batch_size,sample_size,pretrained):
    class_features=get_features(class_data,batch_size,sample_size,pretrained)
    average_feature=torch.mean(class_features,dim=0,keepdim=True)
    dist=dist_matrix(class_features,average_feature).squeeze()
    sorted_inds=torch.argsort(dist,descending=False)
    sorted_inds=sorted_inds[:sample_size]
    return sorted_inds

def update_random(class_data,batch_size,sample_size,pretrained):
    class_size=class_data.size(0)
    random_inds=torch.randperm(class_size)
    return random_inds[:sample_size]

# we try to filter the outliers, filter out 500 points
def update_robust_kmeans(class_data,batch_size,sample_size,pretrained):
    # get the distance to the feature mean and filter out the outliers
    class_features=get_features(class_data,batch_size,sample_size,pretrained)
    average_feature=torch.mean(class_features,dim=0,keepdim=True)
    dist=dist_matrix(class_features,average_feature).squeeze()
    sorted_inds=torch.argsort(dist,descending=False)

    class_size=class_data.size(0)
    candidate_size=int(class_size*9/10)
    candidate_inds=sorted_inds[:candidate_size]
    candidate_features=class_features[candidate_inds]

    _, center_features = kmeans(X=candidate_features, num_clusters=sample_size, distance='euclidean', device=torch.device('cuda:'+str(args.gpu_id)))
    center_features=center_features.cuda()
    dist=dist_matrix(center_features,candidate_features)
    inds=torch.argmin(dist,dim=1)
    final_inds=candidate_inds[inds]

    return final_inds





def update_kmeans(class_data,batch_size,sample_size,pretrained):
    class_features=get_features(class_data,batch_size,sample_size,pretrained)
    _, center_features = kmeans(X=class_features, num_clusters=sample_size, distance='euclidean', device=torch.device('cuda:'+str(args.gpu_id)))
    center_features=center_features.cuda()
    dist=dist_matrix(center_features,class_features)
    inds=torch.argmin(dist,dim=1)
    return inds



# python offline_selection.py --gpu_id 1 --strategy random 
# python offline_selection.py --gpu_id 2 --strategy robust_kmeans

parser = argparse.ArgumentParser()
parser.add_argument('--i', default='raw/cifar10_full.pt', help='input directory')
parser.add_argument('--o', default='none', help='output file')
parser.add_argument('--sample_size',default=10000,type=int)
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--strategy', type=str, default='MOF')
parser.add_argument('--pretrained', type=str, default='cifar100', 
    help='the pretrained model we use as feature extractor')
args = parser.parse_args()


if __name__ == "__main__":

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)
    num_class=10
    sample_per_class=int(args.sample_size/num_class)

    # strategy total_sample_size  pretrained_model for feature extraction
    args.o='cifar10_'+args.strategy+'_'+str(args.sample_size)+'_'+args.pretrained+'.pt'

    print('strategy: %s'%args.strategy)
    print('sample_size: %d'%args.sample_size)
    print('file_name: %s'%args.o)
    print('sample_per_class: %d'%sample_per_class)

    pretrained=pretrained_cifar.cifar_resnet20(pretrained=args.pretrained)
    pretrained.cuda()
    for param in pretrained.parameters():
        param.requires_grad = False
    pretrained.eval()


    x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))
    x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
    x_te = x_te.float().view(x_te.size(0), -1) / 255.0
    y_tr=torch.LongTensor(y_tr)
    y_te=torch.LongTensor(y_te)

    all_data=None
    all_labs=None

    train_data=[]
    test_data=[x_te,y_te]

    for class_id in range(num_class):
        mask=torch.eq(y_tr,class_id)
        class_data=x_tr[mask]
        class_labs=y_tr[mask]
        class_data=class_data.cuda()

        if args.strategy=='MOF':
            inds=update_MOF(class_data,100,sample_per_class,pretrained)
        elif args.strategy=='kmeans':
            inds=update_kmeans(class_data,100,sample_per_class,pretrained)
        elif args.strategy=='random':
            inds=update_random(class_data,100,sample_per_class,pretrained)
        elif args.strategy=='robust_kmeans':
            inds=update_robust_kmeans(class_data,100,sample_per_class,pretrained)

        class_data=class_data[inds].cpu()
        class_labs=class_labs[inds].cpu()

        if all_labs is None:
            all_data=class_data
            all_labs=class_labs
        else:
            all_data=torch.cat((all_data,class_data),dim=0)
            all_labs=torch.cat((all_labs,class_labs),dim=0)
        print('class_id: %d'%class_id)
        print('sampled class size: %d'%inds.size(0))


    random_inds=torch.randperm(args.sample_size)
    all_data=all_data[random_inds]
    all_labs=all_labs[random_inds]

    for i in range(num_class):
        mask=torch.eq(all_labs,i)
        print('class_id: %d   size: %d'%(i,all_labs[mask].size(0)))


    train_data=[all_data,all_labs]
    torch.save([train_data, test_data], args.o)
