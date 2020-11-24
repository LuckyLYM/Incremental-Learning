import numpy as np
import subprocess
import pickle
import torch
import os
import torchvision.datasets as dset


valid= dset.SVHN(root='./', split='test', download=True)
train= dset.SVHN(root='./', split='train', download=True)

train_data=train.data
train_label=train.labels

valid_data=valid.data
valid_label=valid.labels


x_tr = torch.from_numpy(train_data)
y_tr = torch.LongTensor(train_label)
x_te = torch.from_numpy(valid_data)
y_te = torch.LongTensor(valid_label)
torch.save((x_tr, y_tr, x_te, y_te), 'SVHN.pt')
