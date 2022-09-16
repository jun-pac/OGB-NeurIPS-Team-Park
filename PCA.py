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
#from torchmetrics.functional import accuracy as Accuracy
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor
from tqdm import tqdm
from sklearn.decomposition import PCA

# python /users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/PCA.py --dim=129

# You must transform all data including author, papers. It can be done with setup function.
# From now, we just care paper feature.

t0=time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=None)
args = parser.parse_args()

ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)
node_feat_path=osp.join(ROOT,'processed','paper','node_feat.npy')

np.random.seed(42)
train_idx=list(dataset.get_idx_split('train'))
valid_idx=list(dataset.get_idx_split('valid'))
pca_train_idx=train_idx+valid_idx
pca_train_idx.sort()
print("PCA train dataset size :",len(pca_train_idx))

if args.dim==None:
    dim='mle' # Automatic choice of dimensionality
else:
    dim=args.dim # Default 129-dim

print("Loading train dataset...")
train_paper_feat=dataset.paper_feat[pca_train_idx]
print(f"Done! {time.time()-t0}")

print("PCA fit_transform...")
pca = PCA(n_components=dim)
train_paper_feat = pca.fit_transform(train_paper_feat)
dim=train_paper_feat.shape[-1]
del train_paper_feat
print(f"Done! {time.time()-t0} | Dim : {dim}")

print("Loading all dataset...")
all_paper_feat=dataset.paper_feat[:]
print(f"Done! {time.time()-t0}")

print("All PCA transform...")
# Must use 3TB memory node.
principalComponents = pca.transform(all_paper_feat)
print(f"Done! {time.time()-t0}")

dim=principalComponents.shape[-1]
np.save(f"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/PCA_{dim}",principalComponents)
print(f"Save npy file done! {time.time()-t0}")
print("Shape :",principalComponents.shape)