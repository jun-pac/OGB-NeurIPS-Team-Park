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
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor
from tqdm import tqdm
from sampler.sample_china import NeighborSampler
from model import Encoder, Model, drop_feature
import torch.nn as nn
from torch_geometric.utils import dropout_adj


ROOT='/fs/ess/PAS1289'

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )

class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[List[int]],
                 in_memory: bool = False, label_disturb_p: float = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory
        self.label_disturb_p=label_disturb_p

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 4

    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)
        print('load dataset',time.time()-t)
        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        #self.train_idx.share_memory_()
        self.train_label=dataset.paper_label[self.train_idx]
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        #self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        #self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        t1=time.time()
        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,      ############## full_feat.npy
                      mode='r', shape=(N, self.num_features))
        self.x = np.empty((N, self.num_features), dtype=np.float16)
        #self.x = np.zeros((N, self.num_features), dtype=np.float16)
        self.x[:] = x
        self.x = torch.from_numpy(self.x).share_memory_()
        print(f"full_feat loading time : {time.time()-t1:.2f}")
        
        self.y = torch.from_numpy(dataset.all_paper_label)

        #path = f'{dataset.dir}/full_adj_t.pt'
        path=f'{dataset.dir}/full_adj_t.pt'                            ############# full_adj_t.pt
        self.adj_t = torch.load(path)
        one_hot_encoding = np.eye(153)[[int(x) for x in self.train_label]].astype(np.float16)
        self.one_hot_dict={}
        for i in range(len(self.train_idx)):
            self.one_hot_dict[self.train_idx[i]]=one_hot_encoding[i]

        #self.relation_ptr=torch.load(f"{dataset.dir}/relation_ptr.pt") ################# relation_ptr.pt
        self.relation_ptr=torch.load("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/relation_ptr.pt")
        #print(f"Loaded relation_ptr : {self.relation_ptr}")
        # Implement this. # I think we should use another file.
        # Ww have to build self.relation_ptr : Tensor(3*N+1,)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')


    def train_dataloader(self):
        print("call_train_dataloader")
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_train,
                               batch_size=self.batch_size, shuffle=False,
                               num_workers=4, relation_ptr=self.relation_ptr)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=2, relation_ptr=self.relation_ptr)

    def test_dataloader(self):  # Node_idx=self.val_idx??
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=2, relation_ptr=self.relation_ptr)

    def hidden_test_dataloader(self): # This is test-dev. Not test-challenge
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=3, relation_ptr=self.relation_ptr)

    def convert_batch_train(self, batch_size, n_id, adjs):
        x = self.x[n_id]
        append_feat = np.zeros((len(n_id), 153),dtype=np.float16)
        for i in range(batch_size, len(n_id)):
            if (n_id[i] in self.one_hot_dict):
                append_feat[i]=self.one_hot_dict[n_id[i]]
                # Random perturb.
                if(torch.rand(1)<self.label_disturb_p):
                    # Should be same label distribution
                    append_feat[i]=self.one_hot_dict[n_id[int(torch.rand(1)*len(n_id))]] 

        x=torch.from_numpy(np.concatenate((x, append_feat), axis=1)).to(torch.float)
                
        #y = self.y[n_id[:batch_size]].to(torch.long)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])

    def convert_batch_test(self, batch_size, n_id, adjs):
        #t0=time.time()
        x = self.x[n_id]
        append_feat = np.zeros((len(n_id), 153),dtype=np.float16)
        for i in range(batch_size, len(n_id)):
            if (n_id[i] in self.one_hot_dict):
                append_feat[i]=self.one_hot_dict[n_id[i]]
        x=torch.from_numpy(np.concatenate((x, append_feat), axis=1)).to(torch.float)
                
        #y = self.y[n_id[:batch_size]].to(torch.long)
        y = self.y[n_id[:batch_size]].to(torch.long)
        #print(f"CONVERT TEST TIME : {time.time()-t0}")
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])

def to_edge_index(adj_t:SparseTensor):
    edi = adj_t.coo()
    #print(edi[0].shape, edi[1].shape, edi[2].shape)
    edge_index = torch.stack([edi[0],edi[1]])
    return edge_index

def dropout(adj_t, p):
    row, col, edge_attr = adj_t.coo()
    mask = col.new_full((col.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)
    subadj_t = adj_t.masked_select_nnz(mask, layout='coo')
    #subadj_t = subadj_t.set_value(None, layout=None)
    return subadj_t

    """
    row, col, edge_attr = adj_t.coo()
    edge_index = torch.stack([row, col])
    (row, col) , edge_attr = dropout_adj(edge_index, edge_attr, p=drop_edge_rate[0],num_nodes=adj_t.size(0))
    #adj_t = SparseTensor(row = edge_index[0], col = edge_index[1], value = edge_attr)
    edge_type = adj_t.storage.value() == j
                    subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
    #print(adj_t.size(1))
    return torch.stack([row,col,edge_attr])
    """



class GRACE(LightningModule):
    def __init__(self, base_model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, hidden_proj_channels: int, num_relations: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5, tau:float = 0.5,
                 drop_edge_rate = (0.4,0.1), drop_feature_rate=(0.0,0.2)):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_pred = Encoder(in_channels, hidden_channels, out_channels, 
                      base_model=base_model, num_layers=num_layers, num_relations=num_relations, dropout=dropout, heads=heads)
        
        self.encoder_target = Encoder(in_channels, hidden_channels, out_channels, 
                      base_model=base_model, num_layers=num_layers, num_relations=num_relations, dropout=dropout, heads=heads)
        
        self.model = Model(self.encoder_pred, self.encoder_target, hidden_channels, hidden_proj_channels, tau)

        self.drop_edge_rate = drop_edge_rate
        self.drop_feature_rate = drop_feature_rate

        self.train_cnt=0
        self.val_cnt=0
        self.test_cnt=0
        self.batch_idx=0

        self.train_loss_sum = 0
        self.val_loss_sum = 0
        self.test_loss_sum = 0
        self.min_val_loss_sum = 0

        self.tau = tau

        self.val_res=[]
        self.test_res=[]

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        return self.model(x, adjs_t)
        

    def training_step(self, batch, batch_idx: int):
        t0=time.time()
        adjs_t = batch.adjs_t
        edge_index_1 = [dropout(adj_t, self.drop_edge_rate[0]) for adj_t in adjs_t]
        edge_index_2 = [dropout(adj_t, self.drop_edge_rate[1]) for adj_t in adjs_t]
    
        x_1 = drop_feature(batch.x, self.drop_feature_rate[0])
        x_2 = drop_feature(batch.x, self.drop_feature_rate[1])
        z1 = self.model.encoder_pred(x_1, edge_index_1)
        with torch.no_grad():
            z2 = self.model.encoder_target(x_2, edge_index_2)
            z2 = z2.detach()
        
        train_loss = self.model.loss(z1, z2)
        self.train_loss_sum += train_loss.item()
        self.train_cnt+=batch.x.shape[0]
            #print(z1, z2)


        self.log('train_loss', train_loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        if(batch_idx%100==0):
            print('train_loss : '+str(train_loss.item())+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx))
            f_log.write('train_loss : '+str(train_loss.item())+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx))
            f_log.write('\n')
            
        for predict_param, target_param in zip(self.encoder_pred.parameters(), self.encoder_target.parameters()):
            target_param.data = self.tau * target_param.data + (1-self.tau) * predict_param.data
        
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        t0=time.time()
        adjs_t = batch.adjs_t
        
        z1 = self.encoder_pred(batch.x, adjs_t)
        z2 = self.encoder_target(batch.x, batch.adjs_t)
        
        val_loss = self.model.loss(z1, z2)
        self.val_loss_sum += val_loss.item()
        self.val_cnt+=batch.x.shape[0]

        #self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        self.batch_idx=batch_idx
        self.log('val_loss', val_loss.item(), on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        if(batch_idx%50==0):
            print('val_loss : '+str(val_loss.item())+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx))
            f_log.write('val_loss : '+str(val_loss.item())+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx))
            f_log.write('\n')
            #f_log.flush()
        

    def test_step(self, batch, batch_idx: int):
        t0=time.time()
        adjs_t = batch.adjs_t
        
        z1 = self.encoder_pred(batch.x, adjs_t)
        z2 = self.encoder_target(batch.x, batch.adjs_t)
        
        test_loss = self.model.loss(z1, z2)
        self.test_loss_sum += test_loss.item()


        self.test_cnt+=batch.x.shape[0]
        self.test_res.append(z1.cpu().numpy())

        #self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_loss', test_loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if(batch_idx%50==0):
            print('test_loss : '+str(test_loss.item())+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(88092//128))
            f_log.write('test_loss : '+str(test_loss.item())+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(88092//128))
            f_log.write('\n')
            #f_log.flush()
    
    def training_epoch_end(self, outputs) -> None:
        print("Train Epoch end... Loss : "+str(self.train_loss_sum/self.train_cnt))
        f_log.write("Train Epoch end... Loss : "+str(self.train_loss_sum/self.train_cnt))
        f_log.write('\n')
        f_log.flush()
        self.train_loss_sum=0
        self.train_cnt=0





    def validation_epoch_end(self, outputs) -> None:
        print("Validation Epoch end... Loss : "+str(self.val_loss_sum/self.val_cnt))
        f_log.write("Validation Epoch end... Loss : "+str(self.val_loss_sum/self.val_cnt))
        f_log.write('\n')
        #if self.batch_idx>=100 and self.val_loss_sum/self.val_cnt<self.min_val_loss:
        #    self.min_val_loss=self.val_loss_sum/self.val_cnt
        #    self.val_res=np.concatenate(self.val_res)
        #    np.save(f'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Grace/val_embed',self.val_res)
        #    print("Succesfully saved!")
        #    f_log.write("Succesfully saved!\n")
        f_log.flush()
        self.val_res=[]
        self.val_loss_sum=0
        self.val_cnt=0
        self.batch_idx=0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--out_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--in-memory', default=True) # Always have to be true in this code.
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ver', type=int, default=0) # Used in ensemble step.
    parser.add_argument('--label_disturb_p', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.99)
    # Must specify seed everytime.
    # Batchsize, N_source need to be precisely selected, but don't change it for now.

    # python OGB-NeurIPS-Team-Park/relation_NS.py --label_disturb_p=0.1 --batch_size=512

    #CHINA-DEBUG
    #python OGB-NeurIPS-Team-Park/relation_NS.py --label_disturb_p=0.0 --batch_size=512

    t0=time.time()
    args = parser.parse_args()
    #args.sizes = [int(i) for i in args.sizes.split('-')]
    sizes=[[80,20,0],[80,20,10]] # Default behavior
    print(args)
    print(f"Sizes : {sizes}")

    seed_everything(42)
    OROOT = '/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Grace'
    NROOT='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/txtlog' # log file's root.
    name=f'/grace_outsize={args.out_channels}'
    path_log = NROOT+name+'.txt'
    f_log=open(path_log,'w')

    if args.ckpt!=None:
        f_log.write(f"Continue from...{args.ckpt}\n")
        f_log.flush()


    # Dataloader
    datamodule = MAG240M(ROOT, args.batch_size, sizes, args.in_memory, args.label_disturb_p)
    datamodule.setup()   

    # Evaluate
    if True:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        # Ignore previous code
        num = 1
        ckpt='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Grace/logs/Grace/lightning_logs/'+str(num)+'/checkpoints/epoch=0-step=2172.ckpt'
        logdir='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Grace/logs/Grace/lightning_logs/'+str(num)

        trainer = Trainer(resume_from_checkpoint=ckpt,
                          progress_bar_refresh_rate=0) # gpus=args.device,
        
        model = GRACE.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16*8 # initially 16
        datamodule.sizes = [[80,20,0],[80,20,10]] * len(sizes)  # (Almost) no sampling...

        loaders = {"train":datamodule.train_dataloader() , "val":datamodule.val_dataloader()}

        model.eval()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        for name, loader in loaders.items():
            y_infer = []
            for batch in tqdm(loader):
                batch = batch.to(device)
                with torch.no_grad():
                    out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                    y_infer.append(out)
            y_infer = torch.cat(y_infer, dim=0)
            torch.save(y_infer,os.path.join(OROOT,name))