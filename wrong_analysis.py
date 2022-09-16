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

# What aspect?
'''
R-GAT inference analysis
1. Ground truth label distribution
2. Average number of neighborhood (all node VS wrong node)
3. Prediction - Ground truth (the class that is most likely to be wrong?)
4. Number of 'Arxiv' neighborhood (all node VS wrong node)
5. Effect of unlabeled nodes on prediction. (?Perturbation method?)
'''