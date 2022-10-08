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
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential, Softmax
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

start_t=time.time()
Batch_size=1000
max_epoch=100
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
            return self.paper_feat[0],np.nan # Error occur. Do not use Test mode for now. Can we use 0 instead?
        

class LinearModel(LightningModule):
    def __init__(self, input=768, hidden_layer=2048, output=153):
        super().__init__()
        self.l1 = Linear(input, hidden_layer)
        self.l2 = Linear(hidden_layer, output)
        self.acc=Accuracy()
        self.acc_sum=0
        self.cnt=0
    
    def forward(self, x):
        return F.softmax(self.l2(torch.relu(self.l1(x))),dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        loss = F.cross_entropy(y_hat, y)
        self.log('train_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%1000==500):
            print("Train accuracy : "+str(tmp_acc)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write("Train accuracy : "+str(tmp_acc)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        self.log('val_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%200==100):
            print("Validation accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write("Validation accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        self.log('test_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%100==0):
            print("Test accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write("Test accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()

    def training_epoch_end(self, outputs) -> None:
        print("Training epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Training end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def validation_epoch_end(self, outputs) -> None:
        print("Validation epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Validation epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def test_epoch_end(self, outputs) -> None:
        print("Test epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Test epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) #0.001
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


class Linear_Dropout_Model(LightningModule):
    def __init__(self, input=768, hidden_layer=2048, output=153):
        super().__init__()
        self.l1 = Linear(input, hidden_layer)
        self.dropout1 = Dropout(0.5)
        self.l3 = Linear(hidden_layer, output)
        self.acc=Accuracy()
        self.acc_sum=0
        self.cnt=0
    
    def forward(self, x):
        y=self.dropout1(torch.relu(self.l1(x)))
        y=F.softmax(self.l3(y),dim=-1)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        loss = F.cross_entropy(y_hat, y)
        self.log('train_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%1000==500):
            print("Train accuracy : "+str(tmp_acc)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write("Train accuracy : "+str(tmp_acc)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        self.log('val_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%200==100):
            print("Validation accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write("Validation accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        self.log('test_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%100==0):
            print("Test accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write("Test accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()

    def training_epoch_end(self, outputs) -> None:
        print("Training epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Training end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def validation_epoch_end(self, outputs) -> None:
        print("Validation epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Validation epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def test_epoch_end(self, outputs) -> None:
        print("Test epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Test epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) #0.001
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


class Linear_BN_Model(LightningModule):
    def __init__(self, input=768, hidden_layer=2000, output=153):
        super().__init__()
        self.l1 = Linear(input, hidden_layer)
        self.bn1 = BatchNorm1d(hidden_layer)
        self.l2 = Linear(hidden_layer, hidden_layer)
        self.bn2 = BatchNorm1d(hidden_layer)
        self.l3 = Linear(hidden_layer, output)
        self.acc=Accuracy()
        self.acc_sum=0
        self.cnt=0
    
    def forward(self, x):
        y=torch.relu(self.bn1(self.l1(x)))
        y=torch.relu(self.bn2(self.l2(y)))
        y=F.softmax(self.l3(y),dim=-1)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        loss = F.cross_entropy(y_hat, y)
        self.log('train_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%1000==500):
            print("Train accuracy : "+str(tmp_acc)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write("Train accuracy : "+str(tmp_acc)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        self.log('val_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%200==100):
            print("Validation accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write("Validation accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        self.log('test_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%100==0):
            print("Test accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write("Test accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()

    def training_epoch_end(self, outputs) -> None:
        print("Training epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Training end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def validation_epoch_end(self, outputs) -> None:
        print("Validation epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Validation epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def test_epoch_end(self, outputs) -> None:
        print("Test epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Test epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) #0.001
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


class Linear_Deep_BN_Model(LightningModule):
    def __init__(self, input=768, hidden_layer=900, output=153):
        super().__init__()
        self.l1 = Linear(input, hidden_layer)
        self.bn1 = BatchNorm1d(hidden_layer)
        self.l2 = Linear(hidden_layer, hidden_layer)
        self.bn2 = BatchNorm1d(hidden_layer)
        self.l3 = Linear(hidden_layer, hidden_layer)
        self.bn3 = BatchNorm1d(hidden_layer)
        self.l4 = Linear(hidden_layer, output)
        self.acc=Accuracy()
        self.acc_sum=0
        self.cnt=0
    
    def forward(self, x):
        y=torch.relu(self.bn1(self.l1(x)))
        y=torch.relu(self.bn2(self.l2(y)))
        y=torch.relu(self.bn3(self.l3(y)))
        y=F.softmax(self.l3(y),dim=-1)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        loss = F.cross_entropy(y_hat, y)
        self.log('train_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%1000==500):
            print("Train accuracy : "+str(tmp_acc)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write("Train accuracy : "+str(tmp_acc)+"(Loss:"+str(loss.item())+") | Idx : "+str(batch_idx*x.shape[0])+" / 1112392 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        self.log('val_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%200==100):
            print("Validation accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write("Validation accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 138949 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x,y=x.float(),y.type(torch.LongTensor)
        y_hat=self(x)
        tmp_acc=self.acc(y_hat, y).item()
        self.acc_sum+=x.shape[0]*tmp_acc
        self.cnt+=x.shape[0]
        self.log('test_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=False, sync_dist=True)
        if(batch_idx%100==0):
            print("Test accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write("Test accuracy : "+str(tmp_acc)+" | Idx : "+str(batch_idx*x.shape[0])+" / 88092 | Time : "+str(time.time()-t0))
            f_log.write('\n')
            f_log.flush()

    def training_epoch_end(self, outputs) -> None:
        print("Training epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Training end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def validation_epoch_end(self, outputs) -> None:
        print("Validation epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Validation epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def test_epoch_end(self, outputs) -> None:
        print("Test epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write("Test epoch end... Accuracy : "+str(self.acc_sum/self.cnt))
        f_log.write('\n')
        f_log.flush()
        self.acc_sum=0
        self.cnt=0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) #0.001
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


# Total [25]GB of memory needed.
# Start!
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='linear_Deep_BN',
                        choices=['linear', 'linear_Dropout', 'linear_Deep_BN', 'linear_DeepDeep_BN'])
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--hidden', type=int, default=2000)

args = parser.parse_args()
MODEL_NAME=args.model
path_log = NROOT+f'/log_{MODEL_NAME}_{str(args.hidden)}.txt'
f_log=open(path_log,'w+')


print("Reading dataset...")

# Init DataLoader from MNIST Dataset 
train_loader = DataLoader(ArxivSet(mode='train'),batch_size=Batch_size,shuffle=True,num_workers=4)
val_loader = DataLoader(ArxivSet(mode='valid'),batch_size=Batch_size,shuffle=False,num_workers=4)
test_loader = DataLoader(ArxivSet(mode='test-dev'),batch_size=Batch_size,shuffle=False,num_workers=4)
print("Done! : [",time.time()-t0,"]s")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device :",device)


# Train
# Command : python OGB-NeurIPS-Team-Park/linear.py --model=linear_Deep_BN
if not args.evaluate:
    if(args.model=='linear'):
        linear_model=LinearModel(hidden_layer=args.hidden)
    elif(args.model=='linear_Dropout'):
        linear_model=Linear_Dropout_Model(hidden_layer=args.hidden)
    elif(args.model=='linear_Deep_BN'):
        linear_model=Linear_BN_Model(hidden_layer=args.hidden)
    elif(args.model=='linear_DeepDeep_BN'):
        linear_model=Linear_Deep_BN_Model(hidden_layer=args.hidden)
        
    if(args.ckpt is not None):
        checkpoint = torch.load(args.ckpt)
        linear_model.load_state_dict(checkpoint['state_dict'])
        
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=3)
    trainer = Trainer(max_epochs=max_epoch,
        callbacks=[checkpoint_callback],
        default_root_dir=f'/users/PAS1289/oiocha/logs/{MODEL_NAME}',
        progress_bar_refresh_rate=0)
    # Train the model
    trainer.fit(linear_model, train_loader, val_loader)

    # Evaluate
    linear_model.eval()
    #trainer.test(linear_model, test_loader) # Target dosen't exist. So its useless.

    # Submit file
    evaluator = MAG240MEvaluator()
    y_preds = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            x, y = batch
            x,y=x.to(device),y.to(device)
            x,y=x.float(),y.type(torch.LongTensor)
            out = linear_model(x).argmax(dim=-1).cpu()
            y_preds.append(out)
    res = {'y_pred': torch.cat(y_preds, dim=0)}
    evaluator.save_test_submission(res, f'results/{MODEL_NAME}',mode='test-dev')



# Evaluate
# Command : python OGB-NeurIPS-Team-Park/linear.py --model=linear_DeepDeep_BN --evaluate
if args.evaluate:
    dirs = glob.glob(f'/users/PAS1289/oiocha/logs/{MODEL_NAME}/lightning_logs/*')
    version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
    logdir = f'/users/PAS1289/oiocha/logs/{MODEL_NAME}/lightning_logs/version_{version}'
    print(f'Evaluating saved model in {logdir}...')
    ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
    print("CKPT :",ckpt)

    trainer = Trainer(resume_from_checkpoint=ckpt,
                            progress_bar_refresh_rate=0) # gpus=args.device,

    if(args.model=='linear'):
        linear_model=LinearModel(hidden_layer=args.hidden)
    elif(args.model=='linear_Dropout'):
        linear_model=Linear_Dropout_Model(hidden_layer=args.hidden)
    elif(args.model=='linear_Deep_BN'):
        linear_model=Linear_BN_Model(hidden_layer=args.hidden)
    elif(args.model=='linear_DeepDeep_BN'):
        linear_model=Linear_Deep_BN_Model(hidden_layer=args.hidden)
        
    if(args.ckpt is not None):
        checkpoint = torch.load(ckpt)
        linear_model.load_state_dict(checkpoint['state_dict'])

    # 0s
    linear_model.eval()
    evaluator = MAG240MEvaluator()
    y_preds = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            x, y = batch
            x,y=x.to(device),y.to(device)
            x,y=x.float(),y.type(torch.LongTensor)
            out = linear_model(x).argmax(dim=-1).cpu()
            y_preds.append(out)
    res = {'y_pred': torch.cat(y_preds, dim=0)}
    evaluator.save_test_submission(res, f'results/{MODEL_NAME}',mode='test-dev')
