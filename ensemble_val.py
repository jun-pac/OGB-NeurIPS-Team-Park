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
valid_idx=dataset.get_idx_split('valid')
paper_label=dataset.paper_label[valid_idx]

def calculate_val_acc(y_preds):
    cnt=len(paper_label)
    acc_cnt=0
    for i in range(len(paper_label)):
        if(y_preds[i]==paper_label[i]):
            acc_cnt+=1
    return acc_cnt/cnt

dir_list=["/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_0.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_1.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_1.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_2.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_2.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_3.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_4.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_5.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_6.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_6.npy",
"/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/rgat_label_7.npy"]

num_ensemble=len(dir_list)
f_log=open('/fs/scratch/PAS1289/result/rgat_label_ensemble.txt','w+')

summed_activation=np.zeros((138949,153)).astype(np.float16)

# Shape : (138949,153)
for i in range(num_ensemble):
    cur_activation=np.load(dir_list[i]).astype(np.float16)
    y_preds = np.argmax(cur_activation, axis=1)

    # Hard voting
    #cur_activation=np.eye(153)[[int(x) for x in y_preds]].astype(np.float16)

    summed_activation = summed_activation + cur_activation
    temp_acc=calculate_val_acc(y_preds)
    print(f'Validation accuracy for {i}th model: {temp_acc}')
    f_log.write(f'Validation accuracy for {i}th model: {temp_acc}\n')
    

# Maximum prediction
evaluator = MAG240MEvaluator()
y_preds = np.argmax(summed_activation, axis=1)
y_pred = y_preds.astype(np.short)
dir_path=f'results/rgat_label_ensemble'
filename = osp.join(dir_path, 'y_pred_mag240m')
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/ensemble_result",y_preds)
np.savez_compressed(filename, y_pred=y_pred)
total_acc=calculate_val_acc(y_preds)
print(f'Total validation accuracy : {total_acc}')
f_log.write(f'Total validation accuracy : {total_acc}\n')
f_log.flush()

'''
SOFT voting
Validation accuracy for 0th model: 0.6755500219504998
Validation accuracy for 1th model: 0.6823222909124931
Validation accuracy for 2th model: 0.6811995768231509
Validation accuracy for 3th model: 0.6805086758450942
Validation accuracy for 4th model: 0.676197741617428
Validation accuracy for 5th model: 0.6787526358592001
Validation accuracy for 6th model: 0.6833370517240138
Validation accuracy for 7th model: 0.677550756032789
Total validation accuracy : 0.6941683639320901

HARD voting
Validation accuracy for 0th model: 0.6755500219504998
Validation accuracy for 1th model: 0.6823222909124931
Validation accuracy for 2th model: 0.6811995768231509
Validation accuracy for 3th model: 0.6805086758450942
Validation accuracy for 4th model: 0.676197741617428
Validation accuracy for 5th model: 0.6787526358592001
Validation accuracy for 6th model: 0.6833370517240138
Validation accuracy for 7th model: 0.677550756032789
Total validation accuracy : 0.6913112005124182

Optimal
Validation accuracy for 0th model: 0.6755500219504998
Validation accuracy for 1th model: 0.6823222909124931
Validation accuracy for 2th model: 0.6823222909124931
Validation accuracy for 3th model: 0.6811995768231509
Validation accuracy for 4th model: 0.6811995768231509
Validation accuracy for 5th model: 0.6805086758450942
Validation accuracy for 6th model: 0.676197741617428
Validation accuracy for 7th model: 0.6787526358592001
Validation accuracy for 8th model: 0.6833370517240138
Validation accuracy for 9th model: 0.6833370517240138
Validation accuracy for 10th model: 0.677550756032789
Total validation accuracy : 0.6949096431064635
'''

'''
python OGB-NeurIPS-Team-Park/RGAT_label.py --seed=4 --ckpt=logs/rgat_label_4/lightning_logs/version_12935274/checkpoints/epoch=9-step=6339.ckpt --sample_dir=OGB-NeurIPS-Team-Park/Sample_idx/rgat_label_4.npy
python OGB-NeurIPS-Team-Park/RGAT_label.py --seed=3 --ckpt=logs/rgat_label_3/lightning_logs/version_12935160/checkpoints/epoch=14-step=9509.ckpt --sample_dir=OGB-NeurIPS-Team-Park/Sample_idx/rgat_label_3.npy
python OGB-NeurIPS-Team-Park/RGAT_label.py --seed=7 --ckpt=logs/rgat_label_7/lightning_logs/version_12942229/checkpoints/epoch=12-step=8241.ckpt --sample_dir=OGB-NeurIPS-Team-Park/Sample_idx/rgat_label_7.npy
python OGB-NeurIPS-Team-Park/RGAT_label.py --seed=1 --ckpt=logs/rgat_label_1/lightning_logs/version_12933273/checkpoints/epoch=18-step=12045.ckpt --sample_dir=OGB-NeurIPS-Team-Park/Sample_idx/rgat_label_1.npy
python OGB-NeurIPS-Team-Park/RGAT_label.py --seed=1 --ckpt=logs/rgat_label_1/lightning_logs/version_12952969/checkpoints/epoch=0-step=633.ckpt --sample_dir=OGB-NeurIPS-Team-Park/Sample_idx/rgat_label_1.npy
python OGB-NeurIPS-Team-Park/RGAT_label.py --seed=2 --ckpt=logs/rgat_label_2/lightning_logs/version_12933725/checkpoints/epoch=14-step=9509.ckpt --sample_dir=OGB-NeurIPS-Team-Park/Sample_idx/rgat_label_2.npy
'''
