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
node_feat_path=osp.join(ROOT,'processed','paper','node_feat.npy')
node_label_path=osp.join(ROOT,'processed','paper','node_label.npy')
node_year_path=osp.join(ROOT,'processed','paper','node_year.npy')
np.random.seed(42)

print("num papers :",dataset.num_papers) # number of paper nodes
print("num authors :",dataset.num_authors) # number of author nodes
print("num institution :",dataset.num_institutions) # number of institution nodes
print("num paper features :",dataset.num_paper_features) # dimensionality of paper features
print("num classes :",dataset.num_classes) # number of subject area classes

edge_index = dataset.edge_index('author', 'writes', 'paper')
row, col = torch.from_numpy(edge_index)
adj_t = SparseTensor(row=row, col=col, sparse_sizes=(len(row),len(col)), is_sorted=True)
print(f"Loading done! : {time.time()-t0}s")

print(f"len(row) : {len(row)}")
print(f"len(col) : {len(col)}")
print(f"adj_t : {adj_t}")

rowcount=adj_t.storage.rowcount()
print(f"type : {rowcount.dtype}")
print(f"rowcount : {rowcount[:20]}")

