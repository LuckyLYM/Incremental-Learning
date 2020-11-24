import argparse
import os.path
import torch
import pdb
parser = argparse.ArgumentParser()


# preprocess the CIFAR10 dataset

parser.add_argument('--i', default='raw/SVHN.pt', help='input directory')
parser.add_argument('--o', default='SVHN.pt', help='output file')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

# normalize data into the [0,1] range
x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0

cpt = int(10 / args.n_tasks)


# fix the data size, 5000 train, 1000 test
t_data=None
t_lab=None
for class_id in range(10):
    mask=torch.eq(y_tr,class_id)
    train_data=x_tr[mask]
    print('class_id: %d size: %d'%(class_id+1,train_data.size(0)))

    train_data=train_data[:5000]    
    train_labs=y_tr[mask][:5000]
    if t_lab is None:
        t_data=train_data
        t_lab=train_labs
    else:
        t_data=torch.cat((t_data,train_data),dim=0)
        t_lab=torch.cat((t_lab,train_labs),dim=0)



v_data=None
v_lab=None
for class_id in range(10):
    mask=torch.eq(y_te,class_id)
    valid_data=x_te[mask]
    print('class_id: %d size: %d'%(class_id+1,valid_data.size(0)))

    valid_data=valid_data[:500]
    valid_labs=y_te[mask][:500]
    if v_lab is None:
        v_data=valid_data
        v_lab=valid_labs
    else:
        v_data=torch.cat((v_data,valid_data),dim=0)
        v_lab=torch.cat((v_lab,valid_labs),dim=0)




for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((t_lab >= c1) & (t_lab < c2)).nonzero().view(-1)
    i_te = ((v_lab >= c1) & (v_lab < c2)).nonzero().view(-1)
    print('class id: %d'%(t+1))
    print('shape: ',i_tr.shape)

    tasks_tr.append([(c1, c2), t_data[i_tr].clone(), t_lab[i_tr].clone()])
    tasks_te.append([(c1, c2), v_data[i_te].clone(), v_lab[i_te].clone()])

torch.save([tasks_tr, tasks_te], args.o)
