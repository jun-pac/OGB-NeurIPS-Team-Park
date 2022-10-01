
import argparse
from enum import auto
import glob
import os
import os.path as osp
import sys
import time
from typing import List, NamedTuple, Optional
from xmlrpc.client import TRANSPORT_ERROR
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
f_log=open("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/metalog.txt",'w+')

edge_index = dataset.edge_index('author', 'writes', 'paper')
row, col = torch.from_numpy(edge_index) # row for author, col for paper 
author_adj_t = SparseTensor(row=row, col=col, sparse_sizes=(dataset.num_authors, dataset.num_papers), is_sorted=True)

PA_rowptr, PA_colidx, _ =author_adj_t.csc() # Now row for paper, col for author 
PA_colptr, PA_rowidx, _ =author_adj_t.csr() # Same. row for paper, col for author
PA_rowcount=author_adj_t.storage.colcount()
PA_colcount=author_adj_t.storage.rowcount()

print(f"len(PA_rowptr) : {len(PA_rowptr)}, max(PA_rowptr) : {PA_rowptr.max()}")
print(f"len(PA_colidx) : {len(PA_colidx)}, max(PA_colidx) : {PA_colidx.max()}")
print(f"len(PA_colptr) : {len(PA_colptr)}, max(PA_colptr) : {PA_colptr.max()}")
print(f"len(PA_rowidx) : {len(PA_rowidx)}, max(PA_rowidx) : {PA_rowidx.max()}")
print(f"len(PA_rowcount) : {len(PA_rowcount)}, max(PA_rowcount) : {PA_rowcount.max()}")
print(f"len(PA_colcount) : {len(PA_colcount)}, max(PA_colcount) : {PA_colcount.max()}")
print(f"dataset.num_papers : {dataset.num_papers}")
print(f"dataset.num_author : {dataset.num_authors}")
print(f"dataset.num_institution : {dataset.num_institutions}")
'''
len(PA_rowptr) : 121751667, max(PA_rowptr) : 386022720
len(PA_colidx) : 386022720, max(PA_colidx) : 122383111
len(PA_colptr) : 122383113, max(PA_colptr) : 386022720
len(PA_rowidx) : 386022720, max(PA_rowidx) : 121751665
len(PA_rowcount) : 121751666, max(PA_rowcount) : 6760
len(PA_colcount) : 122383112, max(PA_colcount) : 4724
dataset.num_papers : 121751666
dataset.num_author : 122383112
dataset.num_institution : 25721
'''

print(f"Dataload done! {time.time()-t0}")
f_log.write(f"Dataload done! {time.time()-t0}\n")
f_log.flush()

chunk_size=1000000
PAP_sample_size=5
APA_sample_size=4

# Logically, PAP matrix should be symmetric matrix, but stastically benefit of symmetricity is not obvious.

def sample_PAP(idx):
    # task for multi processing
    sampled_row=[]
    sampled_col=[]
    for i in range(idx,min(idx+chunk_size,dataset.num_papers)):
        trow=[] # Should be INT list
        tcol=[] # Should be INT list
        p=[]
        
        # First sample Author without replacement with Author's rowcount distribution.
        for j in range(PA_rowptr[i],PA_rowptr[i+1]):
            p.append(PA_colcount[PA_colidx[j]])
        sampled_A=[]
        if len(p)!=0:
            sampled_A=random.choices(range(PA_rowptr[i].item(),PA_rowptr[i+1].item()), weights=p, k=PAP_sample_size)
            
        # For each sampled author, sample node without replacement.
        for j in PA_colidx[sampled_A]:
            if PA_colcount[j]==0:
                continue
            tcol.append(int(PA_rowidx[random.randint(int(PA_colptr[j]), int(PA_colptr[j+1]-1))]))
        
        # Delete duplicated elements, and node i.
        tcol=set(tcol)
        if i in tcol:
            tcol.remove(i)
        tcol = [*tcol]
        trow=[i]*len(tcol)
        sampled_row+=trow
        sampled_col+=tcol

    if(idx%1000000==0):
        print(f"sample_PAP {idx}th iter... {time.time()-t0}")
        f_log.write(f"sample_PAP {idx}th iter... {time.time()-t0}\n")
        f_log.flush()
    del tcol, trow
    del p, sampled_A
    return np.array([sampled_row, sampled_col])
# Now set all values to 5

def sample_APA(idx):
    # task for multi processing
    sampled_row=[]
    sampled_col=[]
    for i in range(idx,min(idx+chunk_size,dataset.num_authors)):
        trow=[] # Should be INT list
        tcol=[] # Should be INT list
        p=[]
        
        # First sample Author without replacement with Author's rowcount distribution.
        for j in range(PA_colptr[i],PA_colptr[i+1]):
            p.append(PA_rowcount[PA_rowidx[j]])
        sampled_A=[]
        if len(p)!=0:
            sampled_A=random.choices(range(PA_colptr[i],PA_colptr[i+1]), weights=p, k=APA_sample_size)

        # For each sampled author, sample node without replacement.
        for j in PA_rowidx[sampled_A]:
            if PA_rowcount[j]==0:
                continue
            tcol.append(int(PA_colidx[random.randint(int(PA_rowptr[j]), int(PA_rowptr[j+1]-1))]))
        
        # Delete duplicated elements, and node i.
        tcol=set(tcol)
        if i in tcol:
            tcol.remove(i)
        tcol = [*tcol]
        trow=[i]*len(tcol)
        sampled_row+=trow
        sampled_col+=tcol
    if(idx%1000000==0):
        print(f"sample_APA {idx}th iter... {time.time()-t0}")
        f_log.write(f"sample_APA {idx}th iter... {time.time()-t0}\n")
        f_log.flush()
    del tcol, trow
    del p, sampled_A
    return np.array([sampled_row, sampled_col])



core_cnt = 48

# Sample APA     # About 750s
with Pool(core_cnt) as p:
    result = p.map(sample_PAP, range(0,dataset.num_papers,chunk_size))
PAP_result = np.concatenate(result, axis=1)
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/PAP_row_col", PAP_result)
print(f"PAP done : {time.time()-t0}")
f_log.write(f"PAP done : {time.time()-t0} | PAP_result.shape : ({len(PAP_result)},{len(PAP_result[0])})\n")
f_log.flush()

row, col = PAP_result
print(f"max(PAP row) : {row.max().item()}, max(PAP col) : {col.max().item()}")
del PAP_result
del result



# Sample APA    # About 500s
with Pool(core_cnt) as p:
    result = p.map(sample_APA, range(0,dataset.num_authors,chunk_size))
APA_result = np.concatenate(result, axis=1)
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/APA_row_col", APA_result)
print(f"APA done : {time.time()-t0}")
f_log.write(f"APA done : {time.time()-t0} | APA_result.shape : ({len(APA_result)},{len(APA_result[0])})\n")
f_log.flush()

row, col = APA_result
print(f"max(APA row) : {row.max().item()}, max(APA col) : {col.max().item()}")
