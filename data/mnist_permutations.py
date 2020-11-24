import argparse
import os.path
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/', help='input directory')
parser.add_argument('--o', default='mnist_permutations.pt', help='output file')
parser.add_argument('--n_tasks', default=3, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr = torch.load(os.path.join(args.i, 'mnist_train.pt'))
x_te, y_te = torch.load(os.path.join(args.i, 'mnist_test.pt'))
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0
y_tr = y_tr.view(-1).long()
y_te = y_te.view(-1).long()




# we have 10 tasks, in each task we have 10 classes??
# just forget about this one..
for t in range(args.n_tasks):

    # random permutation of the feature
    p = torch.randperm(x_tr.size(1)).long().view(-1)
    
    #let the first task mnist without permutation
    if t==0:
        p=torch.arange(0,28*28)

    # why they do this?? set random permutation here?? weird..
    
    tasks_tr.append(['random permutation', x_tr.index_select(1, p), y_tr])
    tasks_te.append(['random permutation', x_te.index_select(1, p), y_te])

torch.save([tasks_tr, tasks_te], args.o)
