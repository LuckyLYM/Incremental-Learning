# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# download and prepare CIFAR10, CIFAR100, MNIST datasets


import numpy as np
import subprocess
import pickle
import torch
import os

cifar10_path = "cifar-10-python.tar.gz"

if not os.path.exists(cifar10_path):
    subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", shell=True)

subprocess.call("tar xzfv cifar-10-python.tar.gz", shell=True)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



# CIFAR10
x_tr=None
for batch in range(5):#only two batches
    cifar10_train = unpickle('cifar-10-batches-py/data_batch_'+str(batch+1))
    if x_tr is None:
        x_tr = torch.from_numpy(cifar10_train[b'data'])
        y_tr = torch.LongTensor(cifar10_train[b'labels'])
    else:
        x_tr = torch.cat((x_tr,torch.from_numpy(cifar10_train[b'data'])),0)
        y_tr = torch.cat((y_tr,torch.LongTensor(cifar10_train[b'labels'])),0)

cifar10_test = unpickle('cifar-10-batches-py/test_batch')
print("cifar 10 train size is ",y_tr.size(0))

x_te = torch.from_numpy(cifar10_test[b'data'])
y_te = torch.LongTensor(cifar10_test[b'labels'])
torch.save((x_tr, y_tr, x_te, y_te), 'cifar10_full.pt')
