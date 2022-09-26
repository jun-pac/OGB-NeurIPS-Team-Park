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

ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)
valid_idx=dataset.get_idx_split('valid')
paper_label=dataset.paper_label[valid_idx]


ensemble_result=np.load("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/ensemble_result.npy")
degree=np.load("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/paper_degree.npy")

mx=9

y_preds=ensemble_result
t_cnt=[0]*(mx+1) # Number of correct paper whose degree is 0,1,2,3+
f_cnt=[0]*(mx+1) # Number of incorrect paper whose degree is 0,1,2,3+

cnt=len(paper_label)
acc_cnt=0
for i in range(valid_idx.shape[0]):
    if(y_preds[i]==paper_label[i]):
        t_cnt[int(min(degree[valid_idx[i]],mx))]+=1
        acc_cnt+=1
    else:
        f_cnt[int(min(degree[valid_idx[i]],mx))]+=1

print(f"Validation accuracy : {acc_cnt/cnt}")
print()

print("Total node's degree distribution")
for i in range(mx):
    print(f"Degree=={i} : {t_cnt[i]+f_cnt[i]}({(t_cnt[i]+f_cnt[i])/cnt*100:.2f}%), Conditional accuracy : {t_cnt[i]/(t_cnt[i]+f_cnt[i]):.5f}")
print(f"Degree>={mx} : {t_cnt[mx]+f_cnt[mx]}({(t_cnt[mx]+f_cnt[mx])/cnt*100:.2f}%), Conditional accuracy : {t_cnt[mx]/(t_cnt[mx]+f_cnt[mx]):.5f}")
print()

print("Correct node's degree distribution")
for i in range(mx):
    print(f"Degree=={i} : {t_cnt[i]}({(t_cnt[i])/acc_cnt*100:.2f}%)")
print(f"Degree>={mx} : {t_cnt[mx]}({(t_cnt[mx])/acc_cnt*100:.2f}%)")
print()

print("Incorrect node's degree distribution")
for i in range(mx):
    print(f"Degree=={i} : {f_cnt[i]}({(f_cnt[i])/(cnt-acc_cnt)*100:.2f}%)")
print(f"Degree>={mx} : {f_cnt[mx]}({(f_cnt[mx])/(cnt-acc_cnt)*100:.2f}%)")


'''
Validation accuracy : 0.6949096431064635

Total node's degree distribution
Degree==0 : 12274(8.83%), Conditional accuracy : 0.60534
Degree==1 : 4992(3.59%), Conditional accuracy : 0.60998
Degree==2 : 3339(2.40%), Conditional accuracy : 0.64990
Degree==3 : 2754(1.98%), Conditional accuracy : 0.66231
Degree==4 : 2483(1.79%), Conditional accuracy : 0.67499
Degree==5 : 2468(1.78%), Conditional accuracy : 0.69814
Degree==6 : 2582(1.86%), Conditional accuracy : 0.68125
Degree==7 : 2625(1.89%), Conditional accuracy : 0.70552
Degree==8 : 2666(1.92%), Conditional accuracy : 0.70030
Degree>=9 : 102766(73.96%), Conditional accuracy : 0.71240

Correct node's degree distribution
Degree==0 : 7430(7.69%)
Degree==1 : 3045(3.15%)
Degree==2 : 2170(2.25%)
Degree==3 : 1824(1.89%)
Degree==4 : 1676(1.74%)
Degree==5 : 1723(1.78%)
Degree==6 : 1759(1.82%)
Degree==7 : 1852(1.92%)
Degree==8 : 1867(1.93%)
Degree>=9 : 73211(75.82%)

Incorrect node's degree distribution
Degree==0 : 4844(11.43%)
Degree==1 : 1947(4.59%)
Degree==2 : 1169(2.76%)
Degree==3 : 930(2.19%)
Degree==4 : 807(1.90%)
Degree==5 : 745(1.76%)
Degree==6 : 823(1.94%)
Degree==7 : 773(1.82%)
Degree==8 : 799(1.88%)
Degree>=9 : 29555(69.72%)
'''

# What aspect?
'''
R-GAT inference analysis
1. Ground truth label distribution
2. Average number of neighborhood (all node VS wrong node)
3. Prediction - Ground truth (the class that is most likely to be wrong?)
4. Number of 'Arxiv' neighborhood (all node VS wrong node)
5. Effect of unlabeled nodes on prediction. (?Perturbation method?)
'''