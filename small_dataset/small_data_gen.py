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
from ogb.lsc import MAG240MDataset
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
#print(f"adj_t : {adj_t}") # Now it is sparse tensor type
rowptr,col,_=adj_t.csr()
#print(f"rowptr : {rowptr}") # Actually rowptr
#print(f"col : {col}") # Col
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
#print(a<0.8)
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

'''
n_id=torch.from_numpy(np.array([2,3]))
sampled_adj_t, sampled_n_id = sample_adj(adj_t, n_id, 5, replace=True)
print(f"sampled with replacement adj_t : {sampled_adj_t}")
print(f"sampled with replacement n_id : {sampled_n_id}")

sampled_adj_t, sampled_n_id = sample_adj(adj_t, n_id, 5, replace=False)
print(f"sampled without replacement adj_t : {sampled_adj_t}")
print(f"sampled without replacement n_id : {sampled_n_id}")

print(torch.Tensor(3))


np.save("./small_adj",adj_t)


s=set()

s.add(1)
s.add(2)
for i in s:
    print(i,end=' ')
print()
print(s.add(3))
print(s.add(2))
for i in s:
    print(i,end=' ')
print()

print(f"randint : {np.random.randint(0,3)}") # 0,1,2


rowptr,col,_=adj_t.csr()
for i in rowptr:
    print(i)
print(rowptr.shape)

idx=torch.from_numpy(np.array([3,4]))
out_rowptr=torch.empty(idx.numel() + 1, dtype=rowptr.dtype)
print(f"idx.numel() : {idx.numel}")
print(f"rowptr.options() : {rowptr.dtype}")
print(f"out_rowptr : {out_rowptr}")

print(rowptr, rowptr.shape)

n_id=[1,2,3]
t_id=torch.clone(torch.Tensor(n_id).to(dtype=rowptr.dtype))
'''


def python_sample(rowptr: torch.Tensor, col: torch.Tensor, idx: torch.Tensor,num_ppr_neighbors: int,
                num_atr_neighbors: int, num_ins_neighbors: int, relation_ptr: torch.Tensor=None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # replace is always true.
    out_rowptr = torch.empty(idx.numel() + 1, dtype=rowptr.dtype) # rowptr.options()
    out_rowptr[0]=0
    cols=[]
    n_ids=[]
    n_id_map={}

    for n in range(idx.numel()):
        i = int(idx[n])
        cols.append([])
        n_id_map[i] = n
        n_ids.append(i)

    print(f"init n_id_map : {n_id_map}")
    print(f"init n_ids : {n_ids}")

    # Sample begin (always without replacement)
    for i in range(idx.numel()):
        n = int(idx[i])
        row_start = int(rowptr[n])
        ppr_count=int(relation_ptr[3*n+1]-relation_ptr[3*n])
        atr_count=int(relation_ptr[3*n+2]-relation_ptr[3*n+1])
        ins_count=int(relation_ptr[3*n+3]-relation_ptr[3*n+2])
        perm=set()

        # Sample ppr
        if(ppr_count <= num_ppr_neighbors):
            for j in range(ppr_count):
                perm.add(j)
        else :
            for j in range(ppr_count-num_ppr_neighbors, ppr_count):
                temp=np.random.randint(0,j)
                if (not temp in perm):
                    perm.add(temp)
                else:
                    perm.add(j)

        # Sample atr
        if(atr_count <= num_atr_neighbors):
            for j in range(atr_count):
                perm.add(ppr_count+j)
        else :
            for j in range(atr_count-num_atr_neighbors, atr_count):
                temp=np.random.randint(0,j)
                if (not temp in perm):
                    perm.add(ppr_count+temp)
                else:
                    perm.add(ppr_count+j)

        # Sample ins
        if(ins_count <= num_ins_neighbors):
            for j in range(ins_count):
                perm.add(ppr_count+atr_count+j)
        else :
            for j in range(ins_count-num_ins_neighbors, ins_count):
                temp=np.random.randint(0,j)
                if (not temp in perm):
                    perm.add(ppr_count+atr_count+temp)
                else:
                    perm.add(ppr_count+atr_count+j)

        for p in perm:
            e = int(row_start + p)
            c = int(col[e]) # As this is csr format, c is real idx of node.

            if (not c in n_id_map):
                n_id_map[c] = len(n_ids)
                # n_id_map is unordered_map : n_idx's value -> n_idx's internal idx
                # don't increase its size 
                n_ids.append(c)

            cols[i].append((n_id_map[c], e))
            # cols : vector<vector<tuple<int,int>>>, first dimension has len(n_idx) size
            # Store col, e_id information
            # n_id_map[c] is sampled node's pseudo idx, e is sampled node's real idx (order of selected edge in TOTAL edges.) 
            # I just have understood that e_id is real EDGE index, NOT real NODE index.

        out_rowptr[i + 1] = out_rowptr[i] + len(cols[i])
        # Generating new rowptr.
        # Of course it interact with pseudo idx.
        # I think out_rowptr is ptr of out_rowptr
    N = len(n_ids)

    print(f"fin n_id_map : {n_id_map}")
    print(f"fin n_ids : {n_ids}")

    #out_n_id = torch.from_blob(n_ids.data(), {N}, col.options()).clone()
    out_n_id=torch.clone(torch.Tensor(n_ids).to(dtype=col.dtype))

    E = out_rowptr[idx.numel()] # Total size of sampled edges. 
    out_col = torch.empty(E, dtype=col.dtype) # col.options()
    out_e_id = torch.empty(E, dtype=col.dtype) # col.options()

    i = 0
    for col_vec in cols:
        col_vec.sort(key=lambda x:x[0])
        for value in col_vec:
            out_col[i] = value[0] # New node ordering
            out_e_id[i] = value[1] # original edge ordering
            i += 1

    return (out_rowptr, out_col, out_n_id, out_e_id)



rowptr, col, value = adj_t.csr()
print(f"rowptr : {rowptr}")
subset=torch.Tensor([3,7])
print(f"\nsubset : {subset}\n")
relation_ptr=torch.Tensor([0,0,0,0,0,0,0,1,1,1,2,2,3,4,4,5,6,6,7,8,8,9,11,12,13]).to(torch.int64)
rowptr, col, n_id, e_id = python_sample(rowptr, col, subset, 100, 100, 100, relation_ptr)

print(f"sampled rowptr : {rowptr}")
print(f"sampled col : {col}")
print(f"sampled n_id : {n_id}")
print(f"sampled e_id : {e_id}")

# Precision debug
samp=torch.Tensor(1).to(torch.long)
samp[0]=99999999999
print(int(samp[0]))

samp2=torch.Tensor(1).to(torch.float32)
samp2[0]=9999999999
print(int(samp2[0]))

samp3=torch.Tensor(1).to(torch.float32)
samp3[0]=9999999999999999
print(int(samp3[0]))

temp=[999999999999]
temp=torch.Tensor(temp)
print(temp)

temp2=[999999999999]
temp2=torch.Tensor(temp2).to(torch.int64)
print(temp2)

temp3=[999999999999]
t3=torch.Tensor(1).to(torch.int64)
for i in range(len(temp3)):
    t3[i]=temp3[i]
print(t3)

'''
adj_t with value : 
            SparseTensor(row=tensor([2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7]),
             col=tensor([2, 5, 7, 0, 4, 0, 2, 6, 7, 2, 5, 6, 7]),
             val=tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
             size=(8, 8), nnz=13, density=20.31%)

rowptr : tensor([ 0,  0,  0,  1,  3,  5,  7,  9, 13])
col : tensor([2, 5, 7, 0, 4, 0, 2, 6, 7, 2, 5, 6, 7])
value : tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

subset : tensor([3,7])

sampled rowptr : tensor([0, 2, 6])
sampled col : tensor([1, 2, 1, 2, 3, 4]) # pseudo node idx.
sampled n_id : tensor([3, 7, 5, 2, 6]) # mapping pseudo idx-> real idx : 
# (3,7), (3,5), (7,7), (7,5), (7,2), (7,6) are sampled.
sampled e_id : tensor([ 2,  1, 12, 10,  9, 11])
'''

ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)
year=dataset.paper_year


row,col,_=adj_t.coo()
print(row)
print(col)
print(n_id)
print(f"total length : {len(row)}")
cnt1=0
cnt2=0
for i in range(len(row)):
    if(year[row[i]] < year[col[i]]):
        cnt1+=1
    else:
        cnt2+=1
print(f"row<col : {cnt1}, row>=col : {cnt2}")
