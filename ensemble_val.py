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

dir_list=[f"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_0.npy"]
num_ensemble=len(dir_list)

summed_activation=np.load(dir_list[0]).astype(np.float16)

# Shape : (138949,153)
for i in range(1,num_ensemble):
    summed_activation = summed_activation + np.load(dir_list[i]).astype(np.float16)

# Maximum prediction
y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, f'results/{args.model}_label_{seed}',
                                       mode='test-dev')

