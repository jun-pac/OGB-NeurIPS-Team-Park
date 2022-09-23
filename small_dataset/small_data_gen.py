import argparse
import glob
import os
import os.path as osp
import sys
import time
from typing import List, NamedTuple, Optional, Tuple
sys.path.insert(0,'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park')    
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Accuracy
#from torchmetrics.functional import accuracy as Accuracy
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric.utils.sparse import dense_to_sparse

def sample_adj(src: SparseTensor, subset: torch.Tensor, num_neighbors: int,
               replace: bool = False) -> Tuple[SparseTensor, torch.Tensor]:

    rowptr, col, value = src.csr()

    rowptr, col, n_id, e_id = torch.ops.torch_sparse.sample_adj(
        rowptr, col, subset, num_neighbors, replace)

    if value is not None:
        value = value[e_id]

    out = SparseTensor(rowptr=rowptr, row=None, col=col, value=value,
                       sparse_sizes=(subset.size(0), n_id.size(0)),
                       is_sorted=True)

    return out, n_id

seed_everything(1)

N=8
adj=torch.rand((N,N))
adj=adj>0.8
# print(f"adj : {adj}") # just a boolean N*N tensor
s_adj=adj.to_sparse()
# print(f"s_adj : {s_adj}") # tensor, but store matrix as sparse form
g_adj=dense_to_sparse(adj)
# print(f"g_adj : {g_adj}") # tuple of tensor, similar as s_adj
row_col, val= g_adj

adj_t = SparseTensor(row=row_col[0], col=row_col[1], sparse_sizes=(N,N), is_sorted=True)
print(f"adj_t : {adj_t}") # Now it is sparse tensor type
rowptr,col,_=adj_t.csr()
print(f"rowptr : {rowptr}") # Actually rowptr
print(f"col : {col}") # Col
# print(f"_ : {_}") # None

# sampling
num_neighbors=2
rowcount=adj_t.storage.rowcount()
#print(f"rowcount : {rowcount}")
#print(f"rowcount.size(0) : {rowcount.size(0)}")
#print(f"rowcount.size() : {rowcount.size()}")

# subset is not None:
rowptr=rowptr[:-1]

rand=torch.rand((rowcount.size(0),num_neighbors))
#print(f"rand: {rand}")
rand.mul_(rowcount.to(rand.dtype).view(-1,1))
#print(f"After rand.mul_ : {rand}")
rand=rand.to(torch.long)
#print(f"After rand.to : {rand}")
rand.add_(rowptr.view(-1,1))
#print(f"After rand.add_ : {rand}")

#print(f"col[rand] : {col[rand]}")
#print(f"adj_t.storage.value() : {adj_t.storage.value()}")
#print(f"adj_t.nnz() : {adj_t.nnz()}")

a=torch.rand(1)
print(a<0.8)
if a<0.8:
    #print(rowptr[int(torch.rand(1)*len(rowptr))])
    #print(rowptr[int(torch.rand(1)*len(rowptr))])
    pass

adj_t2 = SparseTensor(row=row_col[0], col=row_col[1], value = torch.arange(adj_t.nnz()), sparse_sizes=(N, N)).t()
#print(f"adj_t2 : {adj_t2}")

adj_t = adj_t.set_value(torch.arange(adj_t.nnz()), layout='coo')
print(f"adj_t with value : {adj_t}")

e_id = adj_t.storage.value()
#print(f"e_id : {e_id}")

size = adj_t.sparse_sizes()[::-1]
#print(f"size : {size}")

#print(adj_t.storage.value()==1)

rowptr, col, value = adj_t.csr()
#print(f"rowptr, col, value : \n{rowptr}\n{col}\n{value}")

n_id=torch.from_numpy(np.array([2,3]))
sampled_adj_t, sampled_n_id = sample_adj(adj_t, n_id, 5, replace=True)
print(f"sampled with replacement adj_t : {sampled_adj_t}")
print(f"sampled with replacement n_id : {sampled_n_id}")

sampled_adj_t, sampled_n_id = sample_adj(adj_t, n_id, 5, replace=False)
print(f"sampled without replacement adj_t : {sampled_adj_t}")
print(f"sampled without replacement n_id : {sampled_n_id}")

print(torch.Tensor(3))


np.save("./small_adj",adj_t)
