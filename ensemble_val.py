# This code only conduct soft vote (Note that it is only for same model setting)
# Should save softmax value of validation nodes

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
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer, seed_everything)
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

dir_list=["/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_0.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_1.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_2.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_3.npy"]
num_ensemble=len(dir_list)

summed_activation=np.zeros((138949,153)).astype(np.float16)

# Shape : (138949,153)
for i in range(num_ensemble):
    summed_activation = summed_activation + np.load(dir_list[i]).astype(np.float16)

# Maximum prediction
evaluator = MAG240MEvaluator()
y_preds = np.argmax(summed_activation, axis=1)
y_pred = y_preds.astype(np.short)
dir_path=f'results/rgat_label_ensemble'
filename = osp.join(dir_path, 'y_pred_mag240m')
np.savez_compressed(filename, y_pred=y_pred)


# Calculate accuracy
ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)
valid_idx=dataset.get_idx_split('valid')
paper_label=dataset.paper_label[valid_idx]
cnt=len(paper_label)
acc_cnt=0
for i in range(len(paper_label)):
    if(y_preds[i]==paper_label[i]):
        acc_cnt+=1
print(acc_cnt/cnt)

f_log=open('/fs/scratch/PAS1289/result/rgat_label_ensemble.txt','w+')
f_log.write(f'Validation accuracy : {acc_cnt/cnt}\n')
f_log.flush()