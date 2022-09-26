# Experiment needed : --batch_size (as we don't use GPU), --label_disturb_p, sample_size
# For sample_size : just follow 

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
# Must be always in_memory setup

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
                 in_memory: bool = False, label_disturb_p: float = 0, time_disturb_p: float = 0.2, bit: int=10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory
        self.label_disturb_p=label_disturb_p
        self.time_disturb_p=time_disturb_p
        self.bit=bit

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

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx.share_memory_()
        self.train_label=dataset.paper_label[self.train_idx]
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx.share_memory_()
        self.test_challenge_idx = torch.from_numpy(dataset.get_idx_split('test-challenge'))
        self.test_challenge_idx.share_memory_()
        
        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        t1=time.time()
        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(N, self.num_features))
        self.x = np.empty((N, self.num_features), dtype=np.float16)
        self.x[:] = x
        self.x = torch.from_numpy(self.x).share_memory_()
        print(f"full_feat loading time : {time.time()-t1:.2f}")
        
        self.y = torch.from_numpy(dataset.all_paper_label)

        #path = f'{dataset.dir}/asym_adj_t.pt'
        path='/fs/ess/PAS1289/mag240m_kddcup2021/asym_adj_t.pt'
        self.adj_t = torch.load(path)
        one_hot_encoding = np.eye(153)[[int(x) for x in self.train_label]].astype(np.float16)
        self.one_hot_dict={}
        for i in range(len(self.train_idx)):
            self.one_hot_dict[int(self.train_idx[i])]=one_hot_encoding[i]

        self.relation_ptr=torch.load("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/mono_relation_ptr.pt")
        self.arxiv_idx=set()
        for i in self.train_idx:
            self.arxiv_idx.add(int(i))
        for i in self.val_idx:
            self.arxiv_idx.add(int(i))
        for i in self.test_idx:
            self.arxiv_idx.add(int(i))
        for i in self.test_challenge_idx:
            self.arxiv_idx.add(int(i))

        # Build positional encoding
        self.num_papers=dataset.num_papers
        self.paper_year=dataset.paper_year[:] # load years to memory!
        _idx=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,4,4,5,5,6,7,8,9,9]
        self.year_to_idx=[0]*1985+_idx
        self.positional_encoding=[]
        for i in range(self.bit):
            wave=np.arange(self.bit)
            wave=np.cos((wave-i)*np.pi/10)
            pos_row=torch.from_numpy(wave)
            #print(f"{i}th encoding : {pos_row}")
            self.positional_encoding.append(pos_row)


        print(f'Done! [{time.perf_counter() - t:.2f}s]')


    def train_dataloader(self):
        print("call_train_dataloader")
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_train,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=10, relation_ptr=self.relation_ptr)

    def val_dataloader(self):
        print("call_val_dataloader")
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=10, relation_ptr=self.relation_ptr)

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
        #t0=time.time()
        x = self.x[n_id]
        append_feat = np.zeros((len(n_id), 153),dtype=np.float16)
        append_time = np.zeros((len(n_id), self.bit),dtype=np.float16)
        
        for i in range(batch_size, len(n_id)):
            if (int(n_id[i]) in self.one_hot_dict):
                append_feat[i]=self.one_hot_dict[int(n_id[i])]
                # Random perturb.
                if(torch.rand(1)<self.label_disturb_p):
                    # Should be same label distribution
                    append_feat[i]=self.one_hot_dict[int(self.train_idx[int(torch.rand(1)*len(self.train_idx))])]
        
        for i in range(len(n_id)):
            if (n_id[i] < self.num_papers):
                year=self.paper_year[n_id[i]]
                # Random perturb
                if(torch.rand(1)<self.time_disturb_p):
                    year=max(min(year+int(torch.randn(1)*3),2021),0)
                append_time[i]=self.positional_encoding[self.year_to_idx[year]]
        x=torch.from_numpy(np.concatenate((x, append_feat, append_time), axis=1)).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)

        adj_t,_a,_b=adjs[0]
        row,col,_=adj_t.coo()
        print(f"total length : {len(row)}")
        cnt1=0
        cnt2=0
        for i in range(len(row)):
            if(n_id[row[i]]>=self.num_papers or n_id[col[i]]>=self.num_papers):
                continue
            if(self.paper_year[n_id[row[i]]] < self.paper_year[n_id[col[i]]]):
                cnt1+=1
            else:
                cnt2+=1
        print(f"row<col : {cnt1}, row>=col : {cnt2}")
        f_log.write(f"total length : {len(row)}\n")
        f_log.write(f"row<col : {cnt1}, row>=col : {cnt2}\n")
        f_log.flush()
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


    def convert_batch_test(self, batch_size, n_id, adjs):
        #t0=time.time()
        x = self.x[n_id]
        append_feat = np.zeros((len(n_id), 153),dtype=np.float16)
        append_time = np.zeros((len(n_id), self.bit),dtype=np.float16)
        for i in range(batch_size, len(n_id)):
            if (int(n_id[i]) in self.one_hot_dict):
                append_feat[i]=self.one_hot_dict[int(n_id[i])]
        for i in range(len(n_id)):
            if (n_id[i] < self.num_papers):
                year=self.paper_year[n_id[i]]
                append_time[i]=self.positional_encoding[self.year_to_idx[year]]
            
        x=torch.from_numpy(np.concatenate((x, append_feat, append_time), axis=1)).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)

        adj_t,_a,_b=adjs[0]
        row,col,_=adj_t.coo()
        print(f"total length : {len(row)}")
        cnt1=0
        cnt2=0
        for i in range(len(row)):
            if(n_id[row[i]]>=self.num_papers or n_id[col[i]]>=self.num_papers):
                continue
            if(self.paper_year[n_id[row[i]]] < self.paper_year[n_id[col[i]]]):
                cnt1+=1
            else:
                cnt2+=1
        print(f"row<col : {cnt1}, row>=col : {cnt2}")
        f_log.write(f"total length : {len(row)}\n")
        f_log.write(f"row<col : {cnt1}, row>=col : {cnt2}\n")
        f_log.flush()
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'rgat':
            self.convs.append(
                ModuleList([
                    GATConv(in_channels, hidden_channels // heads, heads,
                            add_self_loops=False) for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        GATConv(hidden_channels, hidden_channels // heads,
                                heads, add_self_loops=False)
                        for _ in range(num_relations)
                    ]))

        elif self.model == 'rgraphsage':
            self.convs.append(
                ModuleList([
                    SAGEConv(in_channels, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.skips.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_acc_sum=0
        self.train_cnt=0
        self.val_acc_sum=0
        self.val_cnt=0
        self.test_acc_sum=0
        self.test_cnt=0
        self.val_res=[]
        self.test_res=[]
        self.max_val_acc=0.
        self.batch_idx=0

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):

            #CHINA-DEBUG
            #print(f"Forward i, adj_t : {i}, {adj_t}")

            x_target = x[:adj_t.size(0)]
            out = self.skips[i](x_target)
            # out tensor is initialized as skip connection of prev. layer(Just Linear layer)
            # And activations added calculated from different relations.
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                subadj_t = subadj_t.set_value(None, layout=None)
                if subadj_t.nnz() > 0:
                    out += self.convs[i][j]((x, x_target), subadj_t)

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        tmp_acc=self.train_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
        self.train_acc_sum+=batch.x.shape[0]*tmp_acc
        self.train_cnt+=batch.x.shape[0]
        self.log('train_acc', tmp_acc, prog_bar=True, on_step=False, on_epoch=True)
        if(batch_idx%100==0):
            print('train_acc : '+str(tmp_acc)+' | loss : '+str(train_loss)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(1112392//args.batch_size))
            f_log.write('train_acc : '+str(tmp_acc)+' | loss : '+str(train_loss)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(1112392//args.batch_size))
            f_log.write('\n')
            f_log.flush()
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        tmp_acc=self.val_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
        self.val_acc_sum+=batch.x.shape[0]*tmp_acc
        self.val_cnt+=batch.x.shape[0]
        self.val_res.append(y_hat.softmax(dim=-1).cpu().numpy())

        #self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        self.batch_idx=batch_idx
        self.log('val_acc', tmp_acc, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        if(batch_idx%50==0):
            print('val_acc : '+str(tmp_acc)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(138949//args.batch_size))
            f_log.write('val_acc : '+str(tmp_acc)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(138949//args.batch_size))
            f_log.write('\n')
            f_log.flush()

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        tmp_acc=self.test_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
        self.test_acc_sum+=batch.x.shape[0]*tmp_acc
        self.test_cnt+=batch.x.shape[0]
        self.test_res.append(y_hat.softmax(dim=-1).cpu().numpy())

        self.log('test_acc', tmp_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if(batch_idx%50==0):
            print('test_acc : '+str(tmp_acc)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(88092//128))
            f_log.write('test_acc : '+str(tmp_acc)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(88092//128))
            f_log.write('\n')
            #f_log.flush()
    
    def training_epoch_end(self, outputs) -> None:
        print("Train Epoch end... Accuracy : "+str(self.train_acc_sum/self.train_cnt))
        f_log.write("Train Epoch end... Accuracy : "+str(self.train_acc_sum/self.train_cnt))
        f_log.write('\n')
        f_log.flush()
        self.train_acc_sum=0
        self.train_cnt=0

    def validation_epoch_end(self, outputs) -> None:
        print("Validation Epoch end... Accuracy : "+str(self.val_acc_sum/self.val_cnt))
        f_log.write("Validation Epoch end... Accuracy : "+str(self.val_acc_sum/self.val_cnt))
        f_log.write('\n')
        if self.batch_idx>=100 and self.val_acc_sum/self.val_cnt>self.max_val_acc:
            self.max_val_acc=self.val_acc_sum/self.val_cnt
            self.val_res=np.concatenate(self.val_res)
            np.save(f'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation'+name, self.val_res)
            print("Succesfully saved!")
            f_log.write("Succesfully saved!\n")
        f_log.flush()
        self.val_res=[]
        self.val_acc_sum=0
        self.val_cnt=0
        self.batch_idx=0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--in-memory', default=True) # Always have to be true in this code.
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--label_disturb_p', type=float, default=0.0)
    parser.add_argument('--time_disturb_p', type=float, default=0.2)
    parser.add_argument('--ver', type=int, default=0) # Used in ensemble step.
    parser.add_argument('--bit', type=int, default=10) # Used in ensemble step.

    # python OGB-NeurIPS-Team-Park/monotonic_NS_debug.py --label_disturb_p=0.2 --batch_size=1024
    
    t0=time.time()
    args = parser.parse_args()
    #args.sizes = [int(i) for i in args.sizes.split('-')]
    sizes=[[40,10,0],[15,10,5]] # Default behavior
    print(args)
    print(f"Sizes : {sizes}")

    seed_everything(42)


    # Initialize log directory
    if args.hidden_channels==1024:
        name=f'/mono-NS_p={args.label_disturb_p}_batch={args.batch_size}'
    else:
        name=f'/mono-NS_p={args.label_disturb_p}_batch={args.batch_size}_hidden={args.hidden_channels}'

    NROOT='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/txtlog' # log file's root.
    path_log = NROOT+name+'.txt'
    f_log=open(path_log,'a')
    if args.ckpt!=None:
        f_log.write(f"Continue from...{args.ckpt}\n")
        f_log.flush()


    # Dataloader
    datamodule = MAG240M(ROOT, args.batch_size, sizes, args.in_memory, args.label_disturb_p, args.time_disturb_p, args.bit)


    # Training
    if not args.evaluate:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        print("Device :",device)
        model = RGNN(args.model, datamodule.num_features+153+args.bit,
                    datamodule.num_classes, args.hidden_channels,
                    datamodule.num_relations, num_layers=len(sizes),
                    dropout=args.dropout)
        if args.ckpt != None:
            checkpoint = torch.load(args.ckpt)
            model.load_state_dict(checkpoint['state_dict'])

        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                           save_top_k=3)
        # tensorboard --logdir=/users/PAS1289/oiocha/logs/rgat/lightning_logs
        # About 1000s... (Without data copying, which consume 2400s)  
        trainer = Trainer(max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir='logs'+name,
                          progress_bar_refresh_rate=0) # gpus=args.device,
        
        trainer.fit(model, datamodule=datamodule)
        

    # Evaluate
    if args.evaluate:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        # Ignore previous code
        ckpt='/users/PAS1289/oiocha/logs/rgat/lightning_logs/version_12845365/checkpoints/epoch=2-step=3260.ckpt'
        logdir='/users/PAS1289/oiocha/logs/rgat/lightning_logs/version_12845365'
        trainer = Trainer(resume_from_checkpoint=ckpt,
                          progress_bar_refresh_rate=0) # gpus=args.device,
        model = RGNN.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16*8 # initially 16
        datamodule.sizes = [[100,100,100],[100,100,100]] * len(sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, 'results'+name, mode='test-dev')

'''
row<col : 1808, row>=col : 123859
total length : 198645
row<col : 1633, row>=col : 133328
total length : 199155
row<col : 1819, row>=col : 118958
row<col : 1677, row>=col : 121219
total length : 218986
total length : 207850
row<col : 1891, row>=col : 134574
total length : 196435
row<col : 1766, row>=col : 123232
row<col : 1712, row>=col : 117430
total length : 209759
total length : 208876
row<col : 1722, row>=col : 126553
total length : 214203
row<col : 2071, row>=col : 127752
total length : 191754
row<col : 1839, row>=col : 127362
row<col : 1625, row>=col : 112083
total length : 202369
total length : 207571
row<col : 1645, row>=col : 121222
total length : 205028
row<col : 1846, row>=col : 126259
'''