
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

t0=time.time() # All process would take ~13404s
ROOT='/fs/ess/PAS1289'
dataset = MAG240MDataset(ROOT)
print(f"dataset.dir : {dataset.dir}")
f_log=open("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/meta_relation.txt","w+")
path_symm='/fs/ess/PAS1289/mag240m_kddcup2021/meta_symm_adj_t.pt'
path_mono ='/fs/ess/PAS1289/mag240m_kddcup2021/meta_mono_adj_t.pt'

print(f"Loading data...")
symm_adj_t = torch.load(path_symm)
mono_adj_t = torch.load(path_mono)
print(f"Done! [{time.time()-t0:.2f}]")




print(f"Build meta relation ptr...")
f_log.write("\n")
f_log.flush()
N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
nums=[dataset.num_papers, dataset.num_papers + dataset.num_authors, N]

rowptr,col,_=symm_adj_t.csr()
meta_relation_ptr=torch.Tensor(3*N+1).to(torch.long)
print(f"col length : {len(col)}")
print(f"rowptr length : {len(rowptr)}")
print(f"N : {N}")

chunk=100

meta_relation_ptr.share_memory_()
meta_relation_ptr[0]=0
print(f"meta_relation_ptr[0] : {meta_relation_ptr[0].dtype}")
print(f"rowptr_dtype : {rowptr[0].dtype}")

def full_task(i):
    row_sz=rowptr[i+1]-rowptr[i]
    j=0
    while (j<row_sz and col[rowptr[i]+j]<nums[0]):
        j+=1
    meta_relation_ptr[3*i+1]=rowptr[i]+j
    
    while (j<row_sz and col[rowptr[i]+j]<nums[1]):
        j+=1
    meta_relation_ptr[3*i+2]=rowptr[i]+j

    meta_relation_ptr[3*i+3]=rowptr[i+1]
    if i%1000000==0:
        print(f"SYMM - {i}th row : {meta_relation_ptr[3*i+1]}, {meta_relation_ptr[3*i+2]}, {meta_relation_ptr[3*i+3]} - time : {time.time()-t0:.2f}")
        f_log.write(f"SYMM - {i}th row : {meta_relation_ptr[3*i+1]}, {meta_relation_ptr[3*i+2]}, {meta_relation_ptr[3*i+3]} - time : {time.time()-t0:.2f}\n")
        f_log.flush()

num = 48
pool = Pool(num)
pool.map(full_task, range(N)) # This would take 2120s
print(f"first ten meta_relation_ptr(before save) : {meta_relation_ptr[:10]}")
f_log.write(f"first ten meta_relation_ptr(before save) : {meta_relation_ptr[:10]}\n")
f_log.flush()
torch.save(meta_relation_ptr, "/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/meta_relation_ptr.pt")
print(f"Done! {time.time()-t0:.2f}")




print(f"Build meta-mono relation ptr...")
f_log.write(f"Build meta-mono relation ptr...\n")
f_log.flush()
N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
nums=[dataset.num_papers, dataset.num_papers + dataset.num_authors, N]

rowptr,col,_=mono_adj_t.csr()
meta_mono_relation_ptr=torch.Tensor(3*N+1).to(torch.long)
print(f"col length : {len(col)}")
print(f"rowptr length : {len(rowptr)}")
print(f"N : {N}")

meta_mono_relation_ptr.share_memory_()
meta_mono_relation_ptr[0]=0
print(f"meta_mono_relation_ptr[0] : {meta_mono_relation_ptr[0].dtype}")
print(f"rowptr_dtype : {rowptr[0].dtype}")

def mono_task(i):
    row_sz=rowptr[i+1]-rowptr[i]
    j=0
    while (j<row_sz and col[rowptr[i]+j]<nums[0]):
        j+=1
    meta_mono_relation_ptr[3*i+1]=rowptr[i]+j

    while (j<row_sz and col[rowptr[i]+j]<nums[1]):
        j+=1
    meta_mono_relation_ptr[3*i+2]=rowptr[i]+j

    meta_mono_relation_ptr[3*i+3]=rowptr[i+1]
    if i%1000000==0:
        print(f"MONO - {i}th row : {meta_mono_relation_ptr[3*i+1]}, {meta_mono_relation_ptr[3*i+2]}, {meta_mono_relation_ptr[3*i+3]} - time : {time.time()-t0:.2f}")
        f_log.write(f"MONO - {i}th row : {meta_mono_relation_ptr[3*i+1]}, {meta_mono_relation_ptr[3*i+2]}, {meta_mono_relation_ptr[3*i+3]} - time : {time.time()-t0:.2f}\n")
        f_log.flush()
num = 48
pool = Pool(num)
pool.map(mono_task, range(N)) # This would take 2120s
print(f"first ten meta_mono_relation_ptr(before save) : {meta_mono_relation_ptr[:10]}")
f_log.write(f"first ten meta_mono_relation_ptr(before save) : {meta_mono_relation_ptr[:10]}\n")
f_log.flush()
torch.save(meta_mono_relation_ptr, "/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/meta_mono_relation_ptr.pt")
print(f"Done! {time.time()-t0:.2f}")