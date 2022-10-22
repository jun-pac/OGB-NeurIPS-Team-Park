import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import os
import os.path as osp
import sys
sys.path.insert(0,'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park')
import torch
from torch_sparse import SparseTensor
from ogb.lsc import MAG240MDataset

t0=time.time()

ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)

train_idx=dataset.get_idx_split('train')
valid_idx=dataset.get_idx_split('valid')
test_idx=dataset.get_idx_split('test-dev')
test_challenge_idx=dataset.get_idx_split('test-challenge')
 
print("Train label distribution")
train_label=dataset.paper_label[train_idx]
train_label_dist=torch.zeros(153).to(torch.float64)
print("?j ",train_label_dist.shape)
for i in train_label:
    train_label_dist[int(i)]+=1
train_label_dist/=len(train_label)

print(f"train_label_dist.dtype : {train_label_dist.dtype}")
print(f"First ten elements : {train_label_dist[:10]}")
print(f"Sum : {train_label_dist.sum()}")
torch.save(train_label_dist,"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/mingi_train_label.pt")


print("Valid label distribution")
valid_label=dataset.paper_label[valid_idx]
valid_label_dist=torch.zeros(153).to(torch.float64)
for i in valid_label:
    valid_label_dist[int(i)]+=1
valid_label_dist/=len(valid_label)

print(f"valid_label_dist.dtype : {valid_label_dist.dtype}")
print(f"First ten elements : {valid_label_dist[:10]}")
print(f"Sum : {valid_label_dist.sum()}")
torch.save(valid_label_dist,"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/mingi_valid_label.pt")