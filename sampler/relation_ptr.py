
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
from sampler.china_NS import NeighborSampler
# Must be always in_memory setup

def get_col_slice(**kw):
    pass

def save_col_slice(**kw):
    pass

ROOT='/fs/ess/PAS1289'

dataset = MAG240MDataset(ROOT)
num_features=768

N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
nums=[dataset.num_papers, dataset.num_papers + dataset.num_authors, N]
relation_ptr=torch.Tensor(3*N+1)

path='/fs/ess/PAS1289/mag240m_kddcup2021/full_adj_t.pt'
adj_t = torch.load(path)
rowptr,col,_=adj_t.csr()

for i in range(rowptr):
    

if not osp.exists(done_flag_path):  # Will take ~3 hours...
    t = time.perf_counter()
    fl=open(log_path,'w')

    print('Generating full feature matrix...')
    fl.write('Generating full feature matrix...')
    fl.write('\n')
    fl.flush()
    node_chunk_size = 100000
    dim_chunk_size = 64
    N = (dataset.num_papers + dataset.num_authors +
            dataset.num_institutions)

    paper_feat = dataset.paper_feat
    x = np.memmap(path, dtype=np.float16, mode='w+',
                    shape=(N, num_features))

    t0=time.time()
    print('Copying paper features...','commit -m 1010 UPD')
    fl.write('Copying paper features...')
    fl.write('\n')
    fl.flush()
    for i in range(0, dataset.num_papers, node_chunk_size):
        if ((i/node_chunk_size)%10==0):
            print("COPY - Progress... :",i,"/",dataset.num_papers,"Consumed time :",time.time()-t0)
            fl.write("COPY - Progress... :"+str(i)+"/"+str(dataset.num_papers)+"| Consumed time :"+str(time.time()-t0))
            fl.write('\n')
            fl.flush()
        j = min(i + node_chunk_size, dataset.num_papers)
        x[i:j] = paper_feat[i:j]
    edge_index = dataset.edge_index('author', 'writes', 'paper')
    row, col = torch.from_numpy(edge_index)
    adj_t = SparseTensor(
        row=row, col=col,
        sparse_sizes=(dataset.num_authors, dataset.num_papers),
        is_sorted=True)
    # Processing 64-dim subfeatures at a time for memory efficiency.
    print('Generating author features...')
    fl.write('Generating author features...')
    fl.write('\n')
    fl.flush()
    t0=time.time()
    for i in range(0, num_features, dim_chunk_size):
        print("GEN_author Progress... ",i,"/",num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
        fl.write("GEN_author Progress... "+str(i)+"/"+str(num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
        fl.write('\n')
        fl.flush()
        j = min(i + dim_chunk_size, num_features)
        inputs = get_col_slice(fl, paper_feat, start_row_idx=0,
                                end_row_idx=dataset.num_papers,
                                start_col_idx=i, end_col_idx=j)
        inputs = torch.from_numpy(inputs)
        outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        del inputs
        save_col_slice(
            fl, x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
            end_row_idx=dataset.num_papers + dataset.num_authors,
            start_col_idx=i, end_col_idx=j)
        del outputs
    edge_index = dataset.edge_index('author', 'institution')
    row, col = torch.from_numpy(edge_index)
    adj_t = SparseTensor(
        row=col, col=row,
        sparse_sizes=(dataset.num_institutions, dataset.num_authors),
        is_sorted=False)
    
    print('Generating institution features...')
    fl.write('Generating institution features...')
    fl.write('\n')
    fl.flush()
    t0=time.time()
    # Processing 64-dim subfeatures at a time for memory efficiency.
    for i in range(0, num_features, dim_chunk_size):
        print("GEN_IN Progress... ",i,"/",num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
        fl.write("GEN_IN Progress... "+str(i)+"/"+str(num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
        fl.write('\n')
        fl.flush()
        j = min(i + dim_chunk_size, num_features)
        inputs = get_col_slice(
            fl, x, start_row_idx=dataset.num_papers,
            end_row_idx=dataset.num_papers + dataset.num_authors,
            start_col_idx=i, end_col_idx=j)
        inputs = torch.from_numpy(inputs)
        outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        del inputs
        save_col_slice(
            fl, x_src=outputs, x_dst=x,
            start_row_idx=dataset.num_papers + dataset.num_authors,
            end_row_idx=N, start_col_idx=i, end_col_idx=j)
        del outputs
    x.flush()
    del x
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    with open(done_flag_path, 'w') as f:
        f.write('done')
    fl.close()
path = f'{dataset.dir}/full_feat.npy'



np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/relation_ptr",relation_ptr)