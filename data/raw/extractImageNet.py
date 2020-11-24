import numpy as np
from PIL import Image
import pickle
import sys
import torch

wnids = list(map(lambda x: x.strip(), open('wnids.txt').readlines()))

'''
data = {}
data['train'] = {}
data['train']['data'] = np.ndarray(shape=(100000, 3, 64, 64), dtype=np.uint8)
data['train']['target'] = np.ndarray(shape=(100000,), dtype=np.uint8)
data['val'] = {}
data['val']['data'] = np.ndarray(shape=(10000, 3, 64, 64), dtype=np.uint8)
data['val']['target'] = np.ndarray(shape=(10000,), dtype=np.uint8)
'''

data = {}
data['train'] = {}
data['train']['data'] = np.ndarray(shape=(100000, 3, 32, 32), dtype=np.uint8)
data['train']['target'] = np.ndarray(shape=(100000,), dtype=np.uint8)
data['val'] = {}
data['val']['data'] = np.ndarray(shape=(10000, 3, 32, 32), dtype=np.uint8)
data['val']['target'] = np.ndarray(shape=(10000,), dtype=np.uint8)


# load train data
for i in range(len(wnids)):
    wnid = wnids[i]
    print("{}: {} / {}".format(wnid, i + 1, len(wnids)))
    for j in range(500):
        path = "train/{0}/{0}_{1}.JPEG".format(wnid, j)
        data['train']['data'][i * 500 + j] = np.asarray(Image.open(path).convert('RGB').resize((32,32))).transpose(2, 0, 1)
        data['train']['target'][i * 500 + j] = i

'''
for i in range(len(wnids)):
    wnid = wnids[i]
    print("{}: {} / {}".format(wnid, i + 1, len(wnids)))
    for j in range(500):
        path = "train/{0}/{0}_{1}.JPEG".format(wnid, j)
        image=Image.open(path).convert('RGB')
        image=image.resize((32,32))
        image=np.asarray(image)
        print(image.shape)
        image=image.transpose(2,0,1)
        print(image.shape)
        if j==0:
            sys.exit()
'''

for i, line in enumerate(list(map(lambda s: s.strip(), open('val/val_annotations.txt')))):
    name, wnid = line.split('\t')[0:2]
    path = "val/{1}/{0}".format(name,wnid)
    # transform images to RGB, CHW format
    # I need to check how my pretrained model implement this part, damn it
    data['val']['data'][i] = np.asarray(Image.open(path).convert('RGB').resize((32,32))).transpose(2, 0, 1)
    data['val']['target'][i] = wnids.index(wnid)




x_tr=torch.from_numpy(data['train']['data'])
y_tr=torch.from_numpy(data['train']['target']).long()
x_te=torch.from_numpy(data['val']['data'])
y_te=torch.from_numpy(data['val']['target']).long()
print("save as torch file")
print(x_tr.shape,y_tr.shape,x_te.shape,y_te.shape)
torch.save((x_tr, y_tr, x_te, y_te), '../tiny32.pt')
