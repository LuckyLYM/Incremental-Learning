import argparse
import os.path
import torch
from utils import noisify
import numpy as np


# python noise.py --dataset cifar10 --n_tasks 5 --noisy yes --noise_rate 0.1


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='none', help='input directory')
parser.add_argument('--i', default='raw/cifar10.pt', help='input directory')
parser.add_argument('--o', default='cifar10.pt', help='output file')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--noisy', default='yes',type=str,help='whether generate noisy tasks')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
args = parser.parse_args()
torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []
args.i=os.path.join('raw',args.dataset+'.pt')
args.o=args.dataset+'.pt'
x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0
cpt = int(10 / args.n_tasks)


if args.noisy=='yes':

    args.o=args.dataset+'_'+args.noise_type+'_'+str(args.noise_rate)+'.pt'
    y_tr=list(np.array(y_tr))
    nb_classes=int(max(y_tr))+1
    train_labels=np.asarray([[y_tr[i]] for i in range(len(y_tr))])
    train_noisy_labels, actual_noise_rate = noisify(train_labels=train_labels, noise_type=args.noise_type, noise_rate=args.noise_rate, random_state=0, nb_classes=nb_classes)
    train_noisy_labels=train_noisy_labels.squeeze()
    y_tr = torch.LongTensor(train_noisy_labels)


for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])


print(args.o)

torch.save([tasks_tr, tasks_te], args.o)
