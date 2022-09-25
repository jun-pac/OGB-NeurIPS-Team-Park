
import argparse
import glob
import os
import os.path as osp
import sys
import time
from typing import List, NamedTuple, Optional
sys.path.insert(0,'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park')    
import numpy as np
import torch
import torch.nn.functional as F
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Accuracy
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor
from tqdm import tqdm
from multiprocessing import Pool, Process, Array

t0=time.time()
ROOT='/fs/ess/PAS1289'
dataset = MAG240MDataset(ROOT)

N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
nums=[dataset.num_papers, dataset.num_papers + dataset.num_authors, N]

print("Loading data...")
path='/fs/ess/PAS1289/mag240m_kddcup2021/full_adj_t.pt'
adj_t = torch.load(path)
rowptr,col,_=adj_t.csr()
print(f"Done! {time.time()-t0}")

relation_ptr=torch.Tensor(3*N+1).to(torch.long)


print(f"col length : {len(col)}")
print(f"rowptr length : {len(rowptr)}")
print(f"N : {N}")

relation_ptr.share_memory_()
relation_ptr[0]=0
print(f"[0] : {relation_ptr[0].dtype}")
print(f"rowptr_dtype : {rowptr[0].dtype}")

def task(i):
    row_sz=rowptr[i+1]-rowptr[i]
    j=0
    while (j<row_sz and col[rowptr[i]+j]<nums[0]):
        j+=1
    relation_ptr[3*i+1]=rowptr[i]+j
    
    while (j<row_sz and col[rowptr[i]+j]<nums[1]):
        j+=1
    relation_ptr[3*i+2]=rowptr[i]+j

    relation_ptr[3*i+3]=rowptr[i+1]
    if i%1000000==0:
        print(f"{i}th row : {relation_ptr[3*i+1]}, {relation_ptr[3*i+2]}, {relation_ptr[3*i+3]} - time : {time.time()-t0:.2f}")

num = 48
pool = Pool(num)
pool.map(task, range(N)) # This would take 2120s

print(f"first ten relation_ptr(before save) : {relation_ptr[:10]}")

torch.save(relation_ptr, "/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/relation_ptr.pt")



print("Loading relation_ptr...")
relation_ptr=torch.load("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/relation_ptr.pt")
print(f"Done! {time.time()-t0:.2f}")

print(f"first ten rowptr : {rowptr[:10]}")
print(f"first ten relation_ptr : {relation_ptr[:10]}")

for i in range(100):
    if(int(rowptr[i]) != int(relation_ptr[3*i])):
        print(f"rowptr : {rowptr[i]}")
        print(f"relation_ptr : {relation_ptr[3*i]}")
        break
    elif(int(relation_ptr[3*i+1])>int(relation_ptr[3*i+2]) or int(relation_ptr[3*i+2])>int(relation_ptr[3*i+3])):
        print(f"Three relation ptr : {int(relation_ptr[3*i+1])}, {int(relation_ptr[3*i+2])}, {int(relation_ptr[3*i+3])}")  
        break

print(f"last element : {rowptr[-1]}, {relation_ptr[-1]}")

'''
122000000th row : 2999310912, 2999310912, 2999310920 - time : 1410.64
123000000th row : 3036098255, 3036098255, 3036098256 - time : 1579.02
first ten relation_ptr(before save) : tensor([0, 0, 2, 2, 4, 6, 6, 6, 8, 8])
Loading relation_ptr...
Done! 1990.59
first ten rowptr : tensor([  0,   2,   6,   8,  48,  51,  53,  62,  86, 116])
first ten relation_ptr : tensor([0, 0, 2, 2, 4, 6, 6, 6, 8, 8])
last element : 3454471824, 3454471824
'''