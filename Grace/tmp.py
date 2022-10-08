import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification
from ogb.lsc import MAG240MDataset, MAG240MEvaluator


ROOT = '/tmp/slurmtmp.13102143'
dataset = MAG240MDataset(ROOT)
print("done")
"""
edge_index = dataset.edge_index('paper', 'cites', 'paper')
print(edge_index[0][:10],edge_index[1][:10])

edge_index = dataset.edge_index('author', 'writes', 'paper')
print(edge_index[0][:10],edge_index[1][:10])
"""

spt = dataset.get_idx_split()
print(spt)