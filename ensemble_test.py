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

ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)

import ogb
print(f"ogb version : {ogb.__version__}")

'''
i_arr=[0]*15+[2]*15+[3]*15+[4]*15 # dir_list_val's partition_idx
dir_list_val=["acua_full_p=0.1_block=0_test_ver=1_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=1_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=1_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=2_test_ver=1_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=1_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=1_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=3_test_ver=1_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=1_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=1_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=4_test_ver=1_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=1_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=1_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=3.npy"
]

dir_list_test=["acua_full_p=0.1_block=0_test_ver=1_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=1_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=1_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=2_test_ver=1_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=1_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=1_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=3_test_ver=1_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=1_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=1_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=4_test_ver=1_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=1_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=1_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=3.npy"
]
'''

i_arr=[0]*12+[1]*12+[2]*12+[3]*12+[4]*12
dir_list_val=[
"acua_full_p=0.1_block=0_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=1_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=1_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=1_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=1_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=1_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=1_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=1_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=1_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=1_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=1_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=1_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=1_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=2_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=3_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=4_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=3.npy"
]


dir_list_test=[
"acua_full_p=0.1_block=0_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=0_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=1_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=1_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=1_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=1_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=1_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=1_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=1_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=1_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=1_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=1_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=1_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=1_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=2_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=2_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=3_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=3_test_ver=3_rank=3.npy",

"acua_full_p=0.1_block=4_test_ver=2.1_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2.1_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2.2_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=2_rank=3.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=1.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=2.npy",
"acua_full_p=0.1_block=4_test_ver=3_rank=3.npy",
]


val_dir='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/'
test_dir='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/test_activation/'
N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
f_log=open('/fs/scratch/PAS1289/result/ensemble_test.txt','w+')

#=============================== Valid ensemble ================================
ori_valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
summed_activation=torch.zeros((138949,153)).type(torch.float64)
L=len(ori_valid_idx)
block_sz=L//5


'''
for ii,dir in enumerate(dir_list_val):
        cur_activation=torch.from_numpy(np.load(val_dir+dir)).type(torch.float64)
        y_preds = torch.argmax(cur_activation, axis=1)
        i=i_arr[ii]

        if(ii==0):
                print(f"Debug - cur_activation.shape : {cur_activation.shape}, cur_activation.dtype : {cur_activation.dtype}")
                print(f"First ten y_preds : {y_preds[:10]}, y_preds.dtype : {y_preds.dtype}")
        
        temp_idx=list(range(block_sz*i, (block_sz*(i+1) if i!=4 else L)))
        if(cur_activation.shape[0]!=len(temp_idx)):
                print(f"Something wrong : {cur_activation.shape[0]}!={len(temp_idx)}")
                break

        for t,idx in enumerate(temp_idx):
                summed_activation[idx]+=cur_activation[t]
        valid_idx=ori_valid_idx[temp_idx]
        paper_label=dataset.paper_label[valid_idx]
        cnt=len(paper_label)
        acc_cnt=0
        for idx in range(len(paper_label)):
                if(y_preds[idx]==paper_label[idx]):
                        acc_cnt+=1
        print(f"{ii}th activation's ({i}th block) accuracy : {acc_cnt/cnt:.5f}")
        f_log.write(f"{ii}th activation's ({i}th block) accuracy : {acc_cnt/cnt:.5f}\n")
        f_log.flush()

print(f"summed_activation[0].sum : {summed_activation[0].sum()}")
print(f"summed_activation.sum : {summed_activation.sum()}")

valid_idx=ori_valid_idx
paper_label=dataset.paper_label[valid_idx]
cnt=len(paper_label)
acc_cnt=0
y_preds = torch.argmax(summed_activation, axis=1)
for i in range(len(paper_label)):
        if(y_preds[i]==paper_label[i]):
                acc_cnt+=1
print(f"Validation summed accuracy : {acc_cnt/cnt:.5f}")
f_log.write(f"Validation summed accuracy : {acc_cnt/cnt:.5f}\n")
f_log.flush()
'''


#=============================== Test ensemble ================================
num_ensemble=len(dir_list_test)
summed_activation=torch.zeros((58726,153)).type(torch.float64)
# Shape : (,153)
for i in range(num_ensemble):
    cur_activation=torch.from_numpy(np.load(test_dir+dir_list_test[i])).type(torch.float64)
    print(f"{i}th sanity check - cur_activation.shape : {cur_activation.shape}, cur_activation.sum() : {cur_activation.sum()}")
    y_preds = torch.argmax(cur_activation, axis=1)
    summed_activation = summed_activation + cur_activation

print(f"Last debug - summed_activation.sum() : {summed_activation.sum()}")

evaluator = MAG240MEvaluator()
y_preds=[summed_activation.argmax(dim=-1).cpu()]
print(f"y_preds[0].shape : {y_preds[0].shape}")
res = {'y_pred': torch.cat(y_preds, dim=0)}
print(f"res['y_pred'].shape : {res['y_pred'].shape}")
print(f"res['y_pred'].dtype : {res['y_pred'].dtype}")
print(f"first ten elements : {res['y_pred'][:10]}")
print(f"maximum element : {torch.max(res['y_pred'])}, minimum element : {torch.min(res['y_pred'])}")
evaluator.save_test_submission(res, '/users/PAS1289/oiocha/results/acua_ensemble', 'test-challenge')