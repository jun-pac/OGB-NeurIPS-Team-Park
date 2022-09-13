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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Accuracy
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor
from tqdm import tqdm
seed_everything(42)

t0=time.time()
# ROOT='/fs/ess/PAS1289'
#ROOT='/tmp' # Copy to tmp : 9:04~9:47
ROOT='/fs/ess/PAS1289'
NROOT='/fs/scratch/PAS1289/data' # log file's root.
path_log = NROOT+'/rgnn_log_linear.txt'
f_log=open(path_log,'w+')
start_t=time.time()
Batch_size=1000
max_epoch=15
DROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = DROOT)

class ArxivSet(Dataset):
    def __init__(self,mode):
        self.mode_idx=dataset.get_idx_split(mode)
        self.paper_feat=dataset.paper_feat[self.mode_idx]
        self.paper_label=dataset.paper_label[self.mode_idx]
        self.len=self.mode_idx.shape[0]
        self.mode=mode
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if(not('test' in self.mode)):
            return self.paper_feat[idx],self.paper_label[idx]
        else:
            return self.paper_feat[0],np.nan
        


class LinearModel(LightningModule):
    def __init__(self, input=768, hidden_layer=2048, output=153):
        super().__init__()
        self.l1 = Linear(input, hidden_layer)
        self.l2 = Linear(hidden_layer, output)
        self.acc=Accuracy()
        self.acc_sum=0
        self.cnt=0
    
    def forward(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat.softmax(dim=-1), y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        loss = F.cross_entropy(y_hat, y)
        if(batch_idx%10==0):
            print("Train accuracy : "+str(self.acc_sum/self.cnt)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write("Train accuracy : "+str(self.acc_sum/self.cnt)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat.softmax(dim=-1), y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        if(batch_idx%10==0):
            print("Validation accuracy : "+str(self.acc_sum/self.cnt)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write("Validation accuracy : "+str(self.acc_sum/self.cnt)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat.softmax(dim=-1), y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        if(batch_idx%10==0):
            print("Test accuracy : "+str(self.acc_sum/self.cnt)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write("Test accuracy : "+str(self.acc_sum/self.cnt)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()

    def training_epoch_end(self, outputs) -> None:
        print("Epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def validation_epoch_end(self, outputs) -> None:
        self.training_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        self.training_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) #0.001
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


# Total []GB of memory needed.
# Start!
# MODEL_NAME='linear'
# MODEL_NAME='linear_Dropout'
MODEL_NAME='linear_Deep_BN'

print("Reading dataset...")

# Init our model
linear_model = LinearModel()

# Init DataLoader from MNIST Dataset 
train_loader = DataLoader(ArxivSet(mode='train'),batch_size=Batch_size,shuffle=True,num_workers=4)
val_loader = DataLoader(ArxivSet(mode='train'),batch_size=Batch_size,shuffle=True,num_workers=4)
test_loader = DataLoader(ArxivSet(mode='test-dev'),batch_size=Batch_size,shuffle=True,num_workers=4)

print("LEN:",len(train_loader))

# Initialize a trainer
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device :",device)
checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=3)
trainer = Trainer(max_epochs=max_epoch,
    callbacks=[checkpoint_callback],
    default_root_dir=f'/users/PAS1289/oiocha/logs/{MODEL_NAME}',
    progress_bar_refresh_rate=0)

print("Done! : [",time.time()-t0,"]s")
# Train the model
trainer.fit(linear_model, train_loader, val_loader)


# Evaluate
dirs = glob.glob(f'/users/PAS1289/oiocha/logs/{MODEL_NAME}/lightning_logs/*')
version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
logdir = f'/users/PAS1289/oiocha/logs/{MODEL_NAME}/lightning_logs/version_{version}'
print(f'Evaluating saved model in {logdir}...')
ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
    
trainer = Trainer(resume_from_checkpoint=ckpt,
                          progress_bar_refresh_rate=0) # gpus=args.device,
model = LinearModel.load_from_checkpoint(
    checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

trainer.test(linear_model, test_loader)

evaluator = MAG240MEvaluator()

# one more time # Since we cant use hidden test dataloader
loader=test_loader
model.eval()
model.to(device)
y_preds = []
for batch in tqdm(loader):
    batch = batch.to(device)
    with torch.no_grad():
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        out = model(x).argmax(dim=-1).cpu()
        y_preds.append(out)
res = {'y_pred': torch.cat(y_preds, dim=0)}
evaluator.save_test_submission(res, f'results/{MODEL_NAME}',mode='test-dev')
