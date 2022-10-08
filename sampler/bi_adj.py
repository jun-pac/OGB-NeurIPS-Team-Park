# Build bi_adj_t, meta_mono_adj_t

import argparse
from enum import auto
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
from torch import Tensor
from torch_sparse import SparseTensor
from tqdm import tqdm
from multiprocessing import Pool, Process, Array
import random

t0=time.time()
ROOT='/fs/ess/PAS1289'
dataset = MAG240MDataset(ROOT)
print(f"dataset.dir : {dataset.dir}")

path='/fs/ess/PAS1289/mag240m_kddcup2021/bi_adj_t.pt'

print(f"Before processing : ")
print(f"dataset.num_papers : {dataset.num_papers}")
print(f"dataset.num_institution : {dataset.num_institutions}")

print(f"Let's begin... {time.time()-t0:.2f}")

# Mono (Asymmetric)     # Would take 330s
# Non trivial behavior
#if not osp.exists(path_mono): 
t = time.perf_counter()

print('Merging MONO adjacency matrices...', flush=True)
edge_index = dataset.edge_index('paper', 'cites', 'paper')
edge_index = torch.from_numpy(edge_index)
row, col = edge_index
rows = [row, col]
cols = [col, row]

edge_index = dataset.edge_index('author', 'writes', 'paper')
row, col = torch.from_numpy(edge_index)
row += dataset.num_papers
rows += [row, col]
cols += [col, row]

edge_index = dataset.edge_index('author', 'institution')
row, col = torch.from_numpy(edge_index)
row += dataset.num_papers
col += dataset.num_papers + dataset.num_authors
rows += [row]
cols += [col]

edge_types = [
    torch.full(x.size(), i, dtype=torch.int8)
    for i, x in enumerate(rows)
]
print(f"largest relation num : {edge_types[-1][0]}")

row = torch.cat(rows, dim=0)
del rows
col = torch.cat(cols, dim=0)
del cols

print(f"max(row) : {row.max().item()}, max(col) : {col.max().item()}")

N = (dataset.num_papers + dataset.num_authors +
        dataset.num_institutions)
print(f"N : {N}")

print(f"argsort... {time.time()-t0:.2f}")
perm = (N * row).add_(col).numpy().argsort()
perm = torch.from_numpy(perm)
row = row[perm]
col = col[perm]

print(f"Build edge_type... {time.time()-t0}")
edge_type = torch.cat(edge_types, dim=0)[perm]
del edge_types

print(f"3 max(APA row) : {row.max().item()}, max(APA col) : {col.max().item()}")
bi_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                            sparse_sizes=(N, N), is_sorted=True)

torch.save(bi_adj_t, path)
print(f'Done! [{time.perf_counter() - t:.2f}s]')
del bi_adj_t

