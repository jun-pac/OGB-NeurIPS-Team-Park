
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


# First, we have to count the maximum number of A-P-A's
#edge_index = dataset.edge_index('author', 'writes', 'paper')
'''
edge_index = dataset.edge_index('author', 'institution') # author(120000000) * ins(25500) matrix
row, col = torch.from_numpy(edge_index)
author_adj_t = SparseTensor(row=row, col=col, is_sorted=True)

rowptr, colidx, _ =author_adj_t.csr()
colptr, rowidx, _ =author_adj_t.csc()
N=len(rowptr)-1

print(f"Done data load. {time.time()-t0}")
print(f"len(rowptr) : {len(rowptr)}")
print(f"len(colptr) : {len(colptr)}")

rowcount=author_adj_t.storage.rowcount()
colcount=author_adj_t.storage.colcount()
PAP_max=torch.zeros(5).to(torch.long) # max, max_idx, min, one count, sum
PA_max=torch.zeros(5).to(torch.long)  # max, max_idx, min, one count, sum
PAP_max[2]=1000000
PA_max[2]=1000000
PAP_max.share_memory_()
PA_max.share_memory_()

print(PAP_max)
print(PA_max)

PAP_num=torch.zeros(N).to(torch.long)
PA_num=torch.zeros(N).to(torch.long)
PAP_num.share_memory_()
PA_num.share_memory_()

chunk=1000000

def task(idx):
    for i in range(idx,min(idx+chunk,N)):
        start=rowptr[i]
        end=rowptr[i+1]
        PA_num[i]=rowcount[i]
        for j in range(start,end):
            author=col[j]
            PAP_num[i]+=colcount[author]
        if PA_num[i]>PA_max[0]:
            PA_max[0]=PA_num[i]
            PA_max[1]=i
        if PA_num[i]<PA_max[2]:
            PA_max[2]=PA_num[i]
        if PA_num[i]==1:
            PA_max[3]+=1
        PA_max[4]+=PA_num[i]
        if PAP_num[i]>PAP_max[0]:
            PAP_max[0]=PAP_num[i]
            PAP_max[1]=i
        if PAP_num[i]<PAP_max[2]:
            PAP_max[2]=PAP_num[i]
        if PAP_num[i]==1:
            PAP_max[3]+=1
        PAP_max[4]+=PAP_num[i]
        
    print(f"{idx}th end... {time.time()-t0}")

num = 48
pool = Pool(num)
pool.map(task, range(0,N,chunk)) # This would take 2120s
#pool.map(task, range(1000))

print(f"Done! ... {time.time()-t0}")

#print(f"PA_num\nFirst ten : {PA_num[:10]}\nLast ten : {PA_num[-10:]}")
#print(f"PAP_num\nFirst ten : {PAP_num[:10]}\nLast ten : {PAP_num[-10:]}")
print(f"PA_num\nFirst ten : {PA_num[:10]}\nLast ten : {PA_num[-10:]}")
print(f"PAP_num\nFirst ten : {PAP_num[:10]}\nLast ten : {PA_num[-10:]}")
print("PAP_num")
print(f"max : {PAP_max[0]}, max idx : {PAP_max[1]}, min : {PAP_max[2]}, one count : {PAP_max[3]}, mean : {PAP_max[4]/dataset.num_papers:.2f}")
print("PA_num")
print(f"max : {PA_max[0]}, max idx : {PA_max[1]}, min : {PA_max[2]}, one count : {PA_max[3]}, mean : {PA_max[4]/dataset.num_papers:.2f}")

#torch.save(PA_num, "/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/PA_num.pt")
#torch.save(PAP_num, "/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/PAP_num.pt")

#PAP_num
#max : 6142224, max idx : 36555, min : 1, one count : 24826748, mean : 256.44
#PA_num
#max : 4724, max idx : 132126, min : 1, one count : 88461913, mean : 3.02

#PA_num
#First ten : tensor([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])
#Last ten : tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
#PAP_num
#First ten : tensor([     0,      0, 104321, 104321,  70071,      0,      0,      0,      0,   3436])
#Last ten : tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
#PAP_num
#max : 3447491, max idx : 118609, min : 0, one count : 745, mean : 12945.85
#PA_num
#max : 78, max idx : 486629, min : 0, one count : 34182447, mean : 0.34

'''


# ==================================== Build Adj =======================================
f_log=open("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/metalog2.txt",'w+')

edge_index = dataset.edge_index('author', 'writes', 'paper')
row, col = torch.from_numpy(edge_index)
author_adj_t = SparseTensor(row=row, col=col, sparse_sizes=(dataset.num_authors, dataset.num_papers), is_sorted=True)

edge_index = dataset.edge_index('author', 'institution')
rowa, cola = torch.from_numpy(edge_index)
ins_adj_t = SparseTensor(row=rowa, col=cola, sparse_sizes=(dataset.num_authors, dataset.num_institutions), is_sorted=True)


PA_rowptr, PA_colidx, _ =author_adj_t.csc()
PA_colptr, PA_rowidx, _ =author_adj_t.csr()
PA_rowcount=author_adj_t.storage.colcount()
PA_colcount=author_adj_t.storage.rowcount()
AI_rowptr, AI_colidx, _ =ins_adj_t.csr()
AI_rowcount=ins_adj_t.storage.rowcount()


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
len(row) : 386022720, len(col) : 386022720
len(PA_rowptr) : 121751667, max(PA_rowptr) : 386022720
len(PA_colidx) : 386022720, max(PA_colidx) : 122383111
len(PA_colptr) : 122383113, max(PA_colptr) : 386022720
len(PA_rowidx) : 386022720, max(PA_rowidx) : 121751665
len(PA_rowcount) : 121751666, max(PA_rowcount) : 337769694
len(PA_colcount) : 122383112, max(PA_colcount) : 4724
dataset.num_papers : 121751666
dataset.num_author : 122383112
dataset.num_institution : 25721
'''

print(f"Dataload done! {time.time()-t0}")
f_log.write(f"Dataload done! {time.time()-t0}\n")
f_log.flush()
'''
for i in range(dataset.num_papers):
    if(PA_rowptr[i]>PA_rowptr[i+1]):
        print(f"{i}th rowptrs : {PA_rowptr[i]}, {PA_rowptr[i+1]}. rowcount : {PA_rowcount[i]}")
        break
print("Clear")
''' 
'''
unique, counts = np.unique(PA_rowcount, return_counts=True)
tuple_list=[]
for i in range(len(unique)):
    tuple_list.append((unique[i],counts[i]))
tuple_list.sort()
for i in range(10):
    print(f"{tuple_list[i][0]} degree : {tuple_list[i][1]}.")
print(f"Largest : {tuple_list[-1][0]} degree : {tuple_list[-1][1]}.")
'''


chunk_size=1000000
sample_size=20

# Logically, PAP matrix should be symmetric matrix but stastically, benefit of symmetricity is not obvious.

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
            sampled_A=random.choices(range(PA_rowptr[i].item(),PA_rowptr[i+1].item()), weights=p, k=sample_size)
            
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

def sample_PAI(idx):
    # task for multi processing
    sampled_row=[]
    sampled_col=[]
    for i in range(idx,min(idx+chunk_size,dataset.num_papers)):
        trow=[] # Should be INT list
        tcol=[] # Should be INT list
        p=[]
        
        # First sample Author without replacement with Author's rowcount distribution.
        for j in range(PA_rowptr[i],PA_rowptr[i+1]):
            p.append(AI_rowcount[PA_colidx[j]])
        sampled_A=[]
        if len(p)!=0:
            sampled_A=random.choices(range(PA_rowptr[i],PA_rowptr[i+1]), weights=p, k=sample_size)

        # For each sampled author, sample node without replacement.
        for j in PA_colidx[sampled_A]:
            if AI_rowcount[j]==0:
                continue
            tcol.append(int(AI_colidx[random.randint(int(AI_rowptr[j]), int(AI_rowptr[j+1]-1))]))
        
        # Delete duplicated elements, and node i.
        tcol=set(tcol)
        tcol = [*tcol]
        trow=[i]*len(tcol)
        sampled_row+=trow
        sampled_col+=tcol
    if(idx%1000000==0):
        print(f"sample_PAI {idx}th iter... {time.time()-t0}")
        f_log.write(f"sample_PAI {idx}th iter... {time.time()-t0}\n")
        f_log.flush()
    del tcol, trow
    del p, sampled_A
    return np.array([sampled_row, sampled_col])



num = 48
'''
with Pool(num) as p:
    result = p.map(sample_PAP, range(0,dataset.num_papers,chunk_size))
PAP_result = np.concatenate(result, axis=1)
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/PAP_row_col2", PAP_result)
print(f"PAP done : {time.time()-t0}")
f_log.write(f"PAP done : {time.time()-t0} | PAP_result.shape : ({len(PAP_result)},{len(PAP_result[0])})\n")
f_log.flush()

row, col = PAP_result
print(f"max(PAP row) : {row.max().item()}, max(PAP col) : {col.max().item()}")

del PAP_result
del result
#row, col = PAP_result
# Append row, col; set value to 5
'''
with Pool(num) as p:
    result = p.map(sample_PAI, range(0,dataset.num_papers,chunk_size))
PAI_result = np.concatenate(result, axis=1)
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/PAI_row_col2", PAI_result)
print(f"PAI done : {time.time()-t0}")
f_log.write(f"PAI done : {time.time()-t0} | PAI_result.shape : ({len(PAI_result)},{len(PAI_result[0])})\n")
f_log.flush()

row, col = PAI_result
print(f"max(PAI row) : {row.max().item()}, max(PAI col) : {col.max().item()}")

# Append row, col; set value to 6
# Append col, row; set value to 7

'''
sample_size=20

sampled_row=[]
sampled_col=[]
cur_len=0
zero_end=0

for i in range(dataset.num_papers):
    trow=[] # Should be INT list
    tcol=[] # Should be INT list
    p=[]
    
    # First sample Author without replacement with Author's rowcount distribution.
    if i<5:
        print(f"PA_rowptr[i], PA_rowptr[i+1], PA_rowptr[dataset.num_papers] : {PA_rowptr[i]}, {PA_rowptr[i+1]}, {PA_rowptr[dataset.num_papers]}")
    for j in range(PA_rowptr[i],PA_rowptr[i+1]):
        p.append(AI_rowcount[PA_colidx[j]])
    sampled_A=[]
    if len(p)!=0:
        sampled_A=random.choices(range(PA_rowptr[i],PA_rowptr[i+1]), weights=p, k=sample_size)
    else:
        zero_end+=1
    # For each sampled author, sample node without replacement.
    for j in PA_colidx[sampled_A]:
        if AI_rowcount[j]==0:
            continue
        tcol.append(int(PA_rowidx[random.randint(int(AI_rowptr[j]), int(AI_rowptr[j+1]-1))]))
    if(i<5):
        print(f"{i}th sampled_A : {sampled_A}")
    if(i<5):
        print(f"{i}th tcol : {tcol}")    
    # Delete duplicated elements, and node i.
    tcol=set(tcol)
    tcol = [*tcol]
    trow=[i]*len(tcol)
    sampled_row+=trow
    sampled_col+=tcol
    cur_len+=len(trow)
    if(i%1000000==0):
        print(f"sample_PAI {i}th iter... {time.time()-t0}, cur_len : {cur_len}, zero_end : {zero_end}")
        f_log.write(f"sample_PAI {i}th iter... {time.time()-t0}, cur_len : {cur_len}, zero_end : {zero_end}\n")
        f_log.flush()

PAI_result = np.array([sampled_row, sampled_col])
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/PAI_row_col", PAI_result)
print(f"PAI done : {time.time()-t0}")
f_log.write(f"PAI done : {time.time()-t0} | PAI_result.shape : ({len(PAI_result)},{len(PAI_result[0])})\n")
f_log.flush()

row, col = PAI_result
print(f"max(PAI row) : {row.max()}, max(PAI col) : {col.max()}")
f_log.write(f"max(PAI row) : {row.max()}, max(PAI col) : {col.max()}\n")
f_log.flush()

sampled_row=[]
sampled_col=[]
cur_len=0

for i in range(dataset.num_papers):
    trow=[] # Should be INT list
    tcol=[] # Should be INT list
    p=[]
    # First sample Author without replacement with Author's rowcount distribution.
    for j in range(PA_rowptr[i],PA_rowptr[i+1]):
        p.append(PA_colcount[PA_colidx[j]])
    sampled_A=[]
    if len(p)!=0:
        sampled_A=random.choices(range(PA_rowptr[i].item(),PA_rowptr[i+1].item()), weights=p, k=sample_size)
        
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
    cur_len+=len(trow)
    sampled_row+=trow
    sampled_col+=tcol

    if(i%100000==0):
        print(f"sample_PAP {i}th iter... {time.time()-t0}, cur_len : {cur_len}")
        f_log.write(f"sample_PAP {i}th iter... {time.time()-t0}, cur_len : {cur_len}\n")
        f_log.flush()
    


PAP_result = np.array([sampled_row, sampled_col])
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/PAP_row_col", PAP_result)
print(f"PAP done : {time.time()-t0}")
f_log.write(f"PAP done : {time.time()-t0} | PAP_result.shape : ({len(PAP_result)},{len(PAP_result[0])})\n")
f_log.flush()

row, col = PAP_result
print(f"max(PAP row) : {row.max()}, max(PAP col) : {col.max()}")
f_log.write(f"max(PAP row) : {row.max()}, max(PAP col) : {col.max()}\n")
f_log.flush()
#row, col = PAP_result
# Append row, col; set value to 5

'''