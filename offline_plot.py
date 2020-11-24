import sys 
sys.path.append("..") 
import os    
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import pickle
import datetime
import argparse
import random
import time
import torch
from kmeans_pytorch import kmeans
from model import pretrained_cifar
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap


# visualize the faetures in low dimension, UMAP

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

def get_features(class_data,batch_size,pretrained):
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

def update_kmeans(class_data,batch_size,sample_size,pretrained):
    class_features=get_features(class_data,batch_size,sample_size,pretrained)
    _, center_features = kmeans(X=class_features, num_clusters=sample_size, distance='euclidean', device=torch.device('cuda:3'))
    center_features=center_features.cuda()
    '''
    print('device of class features: %d'%class_features.get_device())
    print('device of center features: %d'%center_features.get_device())
    print(center_features.shape)
    '''
    dist=dist_matrix(center_features,class_features)
    inds=torch.argmin(dist,dim=1)
    return inds


def update_ICARL(class_data,batch_size,sample_size,pretrained):
    '''
        sum of features is closest to the feature mean. Try this quickly.
    '''
    class_features=get_features(class_data,batch_size,sample_size,pretrained)


def get_inds(strategy,class_id,class_data,batch_size,sample_per_class,pretrained):
    file_name='cifar10_'+strategy+'_'+str(args.sample_size)+'_inds.pt'
    if os.path.exists(file_name):
        inds_dict=torch.load(file_name)
        print('load ',file_name)
        return inds_dict[class_id]
    else:
        print('file not found')


def plot_dataset(data,num_class,inds_dict,class_size,strategy,space,dataset):
    file_name=dataset+'_'+strategy+'_'+space


    # prepare the label_list
    label_list=[]
    for class_id in range(num_class):
        inds=list(inds_dict[class_id].cpu().numpy())
        temp_list=[class_id for i in range(class_size)]
        for i in inds:
            temp_list[i]=num_class
        label_list.extend(temp_list)


    #sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
    sns.set(style='white', context='notebook')
    reducer = umap.UMAP(n_components=2)
    mapper = reducer.fit_transform(data)
    # umap.plot.points(mapper, labels=label_list)

    color_list=sns.color_palette(n_colors=num_class+1)
    plt.scatter(
        mapper[:, 0],
        mapper[:, 1],
        c=[color_list[x] for x in label_list])

    #plt.gca().set_aspect('equal', 'datalim')
    plt.title(file_name, fontsize=24)
    print('save ',file_name)
    plt.savefig(file_name+'.png')



def plot_UMAP(data,inds,class_id,strategy,space,dataset):
    file_name=dataset+'_'+strategy+'_'+space+'_'+str(class_id)


    size=data.shape[0]
    label_list=[0 for i in range(size)]
    for i in inds:
        label_list[i]=1

    #sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
    sns.set(style='white', context='notebook')
    reducer = umap.UMAP(n_components=2,n_neighbors=4990)
    mapper = reducer.fit_transform(data)
    # umap.plot.points(mapper, labels=label_list)

    plt.scatter(
        mapper[:, 0],
        mapper[:, 1],
        c=[sns.color_palette()[x] for x in label_list])

    #plt.gca().set_aspect('equal', 'datalim')
    plt.title(file_name, fontsize=24)
    print('save ',file_name)
    plt.savefig(file_name+'.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', default='raw/cifar10_full.pt', help='input directory')
    parser.add_argument('--o', default='cifar10_MOF.pt', help='output file')
    parser.add_argument('--sample_size',default=10000,type=int)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--strategy', type=str, default='MOF')
    parser.add_argument('--plot_strategy', type=str, default='single')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)

    num_class=10
    sample_per_class=int(args.sample_size/num_class)

    print('strategy: %s'%args.strategy)
    print('sample_size: %d'%args.sample_size)
    print('sample_per_class: %d'%sample_per_class)

    pretrained=pretrained_cifar.cifar_resnet20(pretrained='cifar100')
    pretrained.cuda()
    for param in pretrained.parameters():
        param.requires_grad = False
    pretrained.eval()


    x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))
    x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
    x_te = x_te.float().view(x_te.size(0), -1) / 255.0

    MOF_inds=torch.load('cifar10_MOF_'+str(args.sample_size)+'_inds.pt')
    kmeans_inds=torch.load('cifar10_kmeans_'+str(args.sample_size)+'_inds.pt')

    strategy_list=['MOF','kmeans']
    dict_list=[MOF_inds,kmeans_inds]

    if args.plot_strategy=='all':
        all_data=None
        all_features=None
        for class_id in range(num_class):
            mask=torch.eq(y_tr,class_id)
            class_data=x_tr[mask]
            class_data=class_data.cuda()
            class_features=get_features(class_data,100,pretrained)
            if all_features is None:
                all_features=class_features
                all_data=class_data
            else:
                all_data=torch.cat((all_data,class_data),dim=0)
                all_features=torch.cat((all_features,class_features),dim=0)
        all_data=all_data.cpu().numpy()
        all_features=all_features.cpu().numpy()

        for index,strategy in enumerate(strategy_list):
            print(strategy)
            d=dict_list[index]
            #plot_dataset(data,num_class,inds_dict,class_size,strategy,space,dataset)
            plot_dataset(all_data,num_class,d,5000,strategy,'input','cifar10')
            plot_dataset(all_features,num_class,d,5000,strategy,'feature','cifar10')

    elif args.plot_strategy=='single':
        # single plot, i.e., plot each class
        for class_id in range(1):
            print('class_id ',class_id)
            mask=torch.eq(y_tr,class_id)
            class_data=x_tr[mask]
            class_data=class_data.cuda()
            class_features=get_features(class_data,100,pretrained).cpu().numpy()
            class_data=class_data.cpu().numpy()

            for index,strategy in enumerate(strategy_list):
                inds_dict=dict_list[index]
                print('strategy ',strategy)
                inds=list(inds_dict[class_id].cpu().numpy())
                # plot_UMAP(class_data,inds,class_id+1,strategy,'input','cifar10')
                plot_UMAP(class_features,inds,class_id+1,strategy,'feature','cifar10')






    '''
    # store inds
    for class_id in range(num_class):
        mask=torch.eq(y_tr,class_id)
        class_data=x_tr[mask]
        class_data=class_data.cuda()
        print('begin MOF')
        inds=update_MOF(class_data,100,sample_per_class,pretrained)
        MOF_inds[class_id]=inds
        print('begin kmeans')
        inds=update_kmeans(class_data,100,sample_per_class,pretrained)
        kmeans_inds[class_id]=inds
        print('class_id: %d'%class_id)
        print('sampled class size: %d'%inds.size(0))

    torch.save(MOF_inds, 'cifar10_MOF_'+str(args.sample_size)+'_inds.pt')
    torch.save(kmeans_inds, 'cifar10_kmeans_'+str(args.sample_size)+'_inds.pt')
    '''

