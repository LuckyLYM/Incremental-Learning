import importlib
import pickle
import datetime
import argparse
import random
import time
import os
import numpy as np
import torch.nn as nn
import logging
import torch
from torch.autograd import Variable
from model import pretrained_cifar
from model import common

# training a model with offline data


class AvgrageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() 
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][0].size(0)
    n_outputs = d_te[1].max().item()
    return d_tr, d_te, n_inputs, n_outputs + 1

class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.data= _dataset[0]
        self.labs= _dataset[1]
    def __getitem__(self, index):
        example, target = self.data[index],self.labs[index]
        return example, target
    def __len__(self):
        return self.labs.size(0)

def train(train_queue, model, criterion, optimizer):
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

        optimizer.zero_grad()
        logits= model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, top5.avg

def infer(valid_queue, model, criterion):
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda()

        logits = model(input)
        loss = criterion(logits, target)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, top5.avg


# python offline.py --n_epochs 200 --total_sample_size 10000 --strategy kmeans --gpu_id 0
# python offline.py --n_epochs 200 --total_sample_size 10000 --strategy MOF --gpu_id 1
# python offline.py --n_epochs 200 --total_sample_size 10000 --strategy random --gpu_id 2
# python offline.py --n_epochs 200 --total_sample_size 10000 --strategy robust_kmeans --gpu_id 3

if __name__ == "__main__":
    logger = logging.getLogger('online')
    parser = argparse.ArgumentParser(description='Continuum learning')

    parser.add_argument('--n_epochs', type=int, default=3,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.05,
                        help='SGD learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--gpu_id', type=int, default=3,
                        help='id of gpu we use for training')
    parser.add_argument('--print_log', type=str, default='yes',
                        help='whether print the log?')
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--data_file', default='x.pt')
    parser.add_argument('--total_sample_size',default=10000,type=int)
    parser.add_argument('--strategy',default='kmeans',type=str)
    parser.add_argument('--extractor',default='cifar100',type=str)
    parser.add_argument('--pretrain',default='None',type=str, help='whether use pretrained model')
    args = parser.parse_args()
    args.print_log= True if args.print_log=='yes' else False


    file=args.dataset+'_'+args.strategy+'_'+str(args.total_sample_size)+'_'+args.extractor
    args.data_file=file+'.pt'

    print('data_file: %s'%args.data_file)


    log_path=os.path.join('log',file)
    f=open(log_path,"w")

    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.gpu_id)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_data, valid_data, n_inputs, n_outputs = load_datasets(args)

    print('train size: ',train_data[1].size(0))
    print('test size: ',valid_data[1].size(0))
    print('pretrained model: ',args.pretrain)

    train_data=Custom_Dataset(train_data)
    valid_data=Custom_Dataset(valid_data)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,shuffle=False,pin_memory=True, num_workers=2)

    if args.pretrain=='None':
        model=common.ResNet18(n_outputs)
        model.cuda()
    else:
        model=pretrained_cifar.cifar_resnet20(pretrained=args.pretrain)
        model.cuda()


    optimizer = torch.optim.SGD(
      model.parameters(),
      args.lr,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


    epochs=args.n_epochs
    for epoch in range(epochs):
        prec1, prec5 = train(train_queue, model, criterion, optimizer)
        #print('TRAIN epoch: %d prec1: %f prec5: %f'%(epoch+1,prec1,prec5))
        prec1, prec5= infer(valid_queue, model, criterion)
        print('VALID epoch: %d prec1: %f prec5: %f'%(epoch+1,prec1,prec5))
        f.write('VALID epoch: %d prec1: %f prec5: %f \n'%(epoch+1,prec1,prec5))

    f.close()