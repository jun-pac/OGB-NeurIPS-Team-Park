
import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as TSNE
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

print("h1",time.time()-t0)

print(dataset.num_papers) # number of paper nodes
print(dataset.num_authors) # number of author nodes
print(dataset.num_institutions) # number of institution nodes
print(dataset.num_paper_features) # dimensionality of paper features
print(dataset.num_classes) # number of subject area classes

train_idx=dataset.get_idx_split('train') # (1112392, 768). It takes 80s. 
paper_feat=dataset.paper_feat[train_idx] # this should be run on normal node.
paper_label=dataset.paper_label[train_idx]
paper_year=dataset.paper_year[train_idx]

print(paper_feat.shape)

print("h2",time.time()-t0)