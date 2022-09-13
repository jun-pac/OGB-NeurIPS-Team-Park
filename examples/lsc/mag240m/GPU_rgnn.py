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

# ROOT='/fs/ess/PAS1289'
#ROOT='/tmp' # Copy to tmp : 9:04~9:47
ROOT='/fs/ess/PAS1289'
NROOT='/fs/scratch/PAS1289/data' # log file's root.
#path_log = NROOT+'/rgnn_log_largemem.txt'
path_log = NROOT+'/rgnn_log_largemem_15h.txt'
f_log=open(path_log,'w+')
start_t=time.time()

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
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(N, self.num_features))

        if self.in_memory:
            self.x = np.empty((N, self.num_features), dtype=np.float16)
            self.x[:] = x
            self.x = torch.from_numpy(self.x).share_memory_()
        else:
            self.x = x

        self.y = torch.from_numpy(dataset.all_paper_label)

        #path = f'{dataset.dir}/full_adj_t.pt'
        path='/fs/ess/PAS1289/mag240m_kddcup2021/full_adj_t.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        time0=time.time()
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)
        '''print("Convert batch : "+str(time.time()-time0))
        f_log.write("Convert batch : "+str(time.time()-time0))
        f_log.write('\n')
        f_log.flush()'''
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

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        time0=time.time()
        for i, adj_t in enumerate(adjs_t):
            # adj_t may contain specific layer's sampled neighbors [Sparse tensor]
            # So adjs_t is num_layers*[N*N sparse tensors]
            # But how to differenciate between different relations?
            # adj_t has masked_select method to automatically select different relations.
            # x is activation tensor
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
        '''print("FORWARD TIME : "+str(time.time()-time0))
        f_log.write("FORWARD TIME : "+str(time.time()-time0))
        f_log.write('\n')
        f_log.flush()'''
        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        tmp_acc=self.train_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
        self.train_acc_sum+=batch.x.shape[0]*tmp_acc
        self.train_cnt+=batch.x.shape[0]
        # dictionary?
        # What side effect previous code has??
        # I think train_acc is just Accuracy type. But how logger detect its class and print meaningful information automatically?
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        if(batch_idx%100==0):
            print('train_acc : '+str(self.train_acc(y_hat.softmax(dim=-1), batch.y))+' | loss : '+str(train_loss)+' | time : '+str(time.time()-start_t)+" | batch : "+str(batch_idx)+'/'+str(1112392//1024))
            f_log.write('train_acc : '+str(self.train_acc(y_hat.softmax(dim=-1), batch.y))+' | loss : '+str(train_loss)+' | time : '+str(time.time()-start_t)+" | batch : "+str(batch_idx)+'/'+str(1112392//1024))
            f_log.write('\n')
            #f_log.flush()
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        tmp_acc=self.val_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
        self.val_acc_sum+=batch.x.shape[0]*tmp_acc
        self.val_cnt+=batch.x.shape[0]
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        if(batch_idx%50==0):
            print('val_acc : '+str(self.val_acc(y_hat.softmax(dim=-1), batch.y))+' | time : '+str(time.time()-start_t)+" | batch : "+str(batch_idx)+'/'+str(138949//1024))
            f_log.write('val_acc : '+str(self.val_acc(y_hat.softmax(dim=-1), batch.y))+' | time : '+str(time.time()-start_t)+" | batch : "+str(batch_idx)+'/'+str(138949//1024))
            f_log.write('\n')
            #f_log.flush()

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        tmp_acc=self.test_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
        self.test_acc_sum+=batch.x.shape[0]*tmp_acc
        self.test_cnt+=batch.x.shape[0]
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if(batch_idx%30==0):
            print('test_acc : '+str(self.test_acc(y_hat.softmax(dim=-1), batch.y))+' | time : '+str(time.time()-start_t)+" | batch : "+str(batch_idx)+'/'+str(88092//128))
            f_log.write('test_acc : '+str(self.test_acc(y_hat.softmax(dim=-1), batch.y))+' | time : '+str(time.time()-start_t)+" | batch : "+str(batch_idx)+'/'+str(88092//128))
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
        f_log.flush()
        self.val_acc_sum=0
        self.val_cnt=0

    def test_epoch_end(self, outputs) -> None:
        print("Test Epoch end... Accuracy : "+str(self.test_acc_sum/self.test_cnt))
        f_log.write("Test Epoch end... Accuracy : "+str(self.test_acc_sum/self.test_cnt))
        f_log.write('\n')
        f_log.flush()
        self.test_acc_sum=0
        self.test_cnt=0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--in-memory', default=True) # action='store_true'
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    # --ckpt = "/users/PAS1289/oiocha/logs/rgat/lightning_logs/version_12871851/checkpoints/epoch=0-step=1086.ckpt"
    # Val accuracy : 0.6410
     
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)
    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)

    if not args.evaluate:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        print("Device :",device)
        model = RGNN(args.model, datamodule.num_features,
                    datamodule.num_classes, args.hidden_channels,
                    datamodule.num_relations, num_layers=len(args.sizes),
                    dropout=args.dropout)
        if args.ckpt is not None:
            checkpoint = torch.load(args.ckpt)
            model.load_state_dict(checkpoint['state_dict'])

        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                           save_top_k=3)
        # tensorboard --logdir=/users/PAS1289/oiocha/logs/rgat/lightning_logs
        # About 1000s... (Without data copying, which consume 2400s)  
        trainer = Trainer(max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs/{args.model}',
                          progress_bar_refresh_rate=0) # gpus=args.device,
        
        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        '''
        dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
        print("dirs :",dirs)
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
        print("ckpt :",ckpt)
        '''
        # Ignore previous code
        ckpt='/users/PAS1289/oiocha/logs/rgat/lightning_logs/version_12845365/checkpoints/epoch=2-step=3260.ckpt'
        logdir='/users/PAS1289/oiocha/logs/rgat/lightning_logs/version_12845365'
        trainer = Trainer(resume_from_checkpoint=ckpt,
                          progress_bar_refresh_rate=0) # gpus=args.device,
        model = RGNN.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16*8 # initially 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

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
        evaluator.save_test_submission(res, f'results/{args.model}',
                                       mode='test-dev')


'''
In Largemem node...
Memory usage : 510GB
Prepare : 1300s
100*1024 forward : 700s
1 epoch : 8000s (2h 20m) (With validation)
'''

