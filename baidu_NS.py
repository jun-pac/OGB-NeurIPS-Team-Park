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
from torch_geometric.data import NeighborSampler
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


def get_col_slice(fl, x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    print("get_col_slice")
    fl.write("get_col_slice")
    fl.write('\n')
    fl.flush()
    t0=time.time()
    for i in range(start_row_idx, end_row_idx, chunk):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)

def save_col_slice(fl, x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    print("save_col_slice")
    fl.write("save_col_slice")
    fl.write('\n')
    fl.flush()
    t0=time.time()
    for i in range(0, end_row_idx - start_row_idx, chunk):
        if((i/chunk)%10==0):
            print("SAVE - Sub routine...",i, "/",(end_row_idx - start_row_idx),"| time :",time.time()-t0)
            fl.write("SAVE - Sub routine..."+str(i)+"/"+str(end_row_idx - start_row_idx)+"| time :"+str(time.time()-t0))
            fl.write('\n')
            fl.flush()
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
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
        dataset = MAG240MDataset(self.data_dir)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_adj_t.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type, sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        #path = f'{dataset.dir}/full_feat.npy'
        #done_flag_path = f'{dataset.dir}/full_feat_done.txt'
        #log_path = f'{dataset.dir}/rgnn_log.txt'
        NEWROOT='/fs/scratch/PAS1289/data'
        path = NEWROOT+'/full_feat.npy'
        done_flag_path = NEWROOT+'/full_feat_done.txt'
        log_path = NEWROOT+'/rgnn_log.txt'

        if not osp.exists(done_flag_path):  # Will take ~3 hours...
            t = time.perf_counter()
            fl=open(log_path,'w')

            print('Generating full feature matrix...')
            fl.write('Generating full feature matrix...')
            fl.write('\n')
            fl.flush()
            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))

            t0=time.time()
            print('Copying paper features...','commit -m 1010 UPD')
            fl.write('Copying paper features...')
            fl.write('\n')
            fl.flush()
            for i in range(0, dataset.num_papers, node_chunk_size):
                if ((i/node_chunk_size)%10==0):
                    print("COPY - Progress... :",i,"/",dataset.num_papers,"Consumed time :",time.time()-t0)
                    fl.write("COPY - Progress... :"+str(i)+"/"+str(dataset.num_papers)+"| Consumed time :"+str(time.time()-t0))
                    fl.write('\n')
                    fl.flush()
                j = min(i + node_chunk_size, dataset.num_papers)
                x[i:j] = paper_feat[i:j]
            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=row, col=col,
                sparse_sizes=(dataset.num_authors, dataset.num_papers),
                is_sorted=True)
            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            fl.write('Generating author features...')
            fl.write('\n')
            fl.flush()
            t0=time.time()
            for i in range(0, self.num_features, dim_chunk_size):
                print("GEN_author Progress... ",i,"/",self.num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
                fl.write("GEN_author Progress... "+str(i)+"/"+str(self.num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
                fl.write('\n')
                fl.flush()
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(fl, paper_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    fl, x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs
            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=col, col=row,
                sparse_sizes=(dataset.num_institutions, dataset.num_authors),
                is_sorted=False)
            
            print('Generating institution features...')
            fl.write('Generating institution features...')
            fl.write('\n')
            fl.flush()
            t0=time.time()
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in range(0, self.num_features, dim_chunk_size):
                print("GEN_IN Progress... ",i,"/",self.num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
                fl.write("GEN_IN Progress... "+str(i)+"/"+str(self.num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
                fl.write('\n')
                fl.flush()
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(
                    fl, x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    fl, x_src=outputs, x_dst=x,
                    start_row_idx=dataset.num_papers + dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs
            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')
            fl.close()
        path = f'{dataset.dir}/full_feat.npy'
    
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
        one_hot_encoding = np.eye(153)[[int(x) for x in self.train_label]].astype(np.float16)
        self.one_hot_dict={}
        for i in range(len(self.train_idx)):
            self.one_hot_dict[self.train_idx[i]]=one_hot_encoding[i]

        print(f'Done! [{time.perf_counter() - t:.2f}s]')


    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_train,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Node_idx=self.val_idx??
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self): # This is test-dev. Not test-challenge
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch_train(self, batch_size, n_id, adjs, initial_batch):
        x = self.x[n_id]
        append_feat = np.zeros((len(n_id), 153),dtype=np.float16)
        for i in range(len(n_id)):
            if (n_id[i] in self.one_hot_dict) and (n_id[i] not in initial_batch):
                append_feat[i]=self.one_hot_dict[n_id[i]]
                # Random perturb.
                if(torch.rand(1)<self.label_disturb_p):
                    # Should be same label distribution
                    append_feat[i]=self.one_hot_dict[n_id[int(torch.rand(1)*len(n_id))]] 

        x=torch.from_numpy(np.concatenate((x, append_feat), axis=1)).to(torch.float)
                
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])

    def convert_batch_test(self, batch_size, n_id, adjs, initial_batch):
        #t0=time.time()
        x = self.x[n_id]
        append_feat = np.zeros((len(n_id), 153),dtype=np.float16)
        for i in range(len(n_id)):
            if (n_id[i] in self.one_hot_dict) and (n_id[i] not in initial_batch):
                append_feat[i]=self.one_hot_dict[n_id[i]]
        x=torch.from_numpy(np.concatenate((x, append_feat), axis=1)).to(torch.float)
                
        y = self.y[n_id[:batch_size]].to(torch.long)
        #print(f"CONVERT TEST TIME : {time.time()-t0}")
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

        #CHINA-DEBUG
        #print(f"Training step - y_hat, batch.y : {y_hat}, {batch.y}")
        #print(f"Sizes - y_hat, batch.y : {y_hat.shape}, {batch.y.shape}")
        
        train_loss = F.cross_entropy(y_hat, batch.y)
        tmp_acc=self.train_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
        self.train_acc_sum+=batch.x.shape[0]*tmp_acc
        self.train_cnt+=batch.x.shape[0]
        # What side effect previous code has??
        # I think train_acc is just Accuracy type. But how logger detect its class and print meaningful information automatically?
        #self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', tmp_acc, prog_bar=True, on_step=False, on_epoch=True)
        if(batch_idx%100==0):
            print('train_acc : '+str(tmp_acc)+' | loss : '+str(train_loss)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(1112392//1024))
            f_log.write('train_acc : '+str(tmp_acc)+' | loss : '+str(train_loss)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(1112392//1024))
            f_log.write('\n')
            #f_log.flush()
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
            print('val_acc : '+str(tmp_acc)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(138949//1024))
            f_log.write('val_acc : '+str(tmp_acc)+' | time : '+str(time.time()-t0)+" | batch : "+str(batch_idx)+'/'+str(138949//1024))
            f_log.write('\n')
            #f_log.flush()

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        tmp_acc=self.test_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
        self.test_acc_sum+=batch.x.shape[0]*tmp_acc
        self.test_cnt+=batch.x.shape[0]
        self.test_res.append(y_hat.softmax(dim=-1).cpu().numpy())

        #self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
            np.save(f'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/baidu-NS_p={args.label_disturb_p}_batch={args.batch_size}',self.val_res)
            print("Succesfully saved!")
            f_log.write("Succesfully saved!\n")
        f_log.flush()
        self.val_res=[]
        self.val_acc_sum=0
        self.val_cnt=0
        self.batch_idx=0

    '''
    def test_epoch_end(self, outputs) -> None:
        print("Test Epoch end... Accuracy : "+str(self.test_acc_sum/self.test_cnt))
        f_log.write("Test Epoch end... Accuracy : "+str(self.test_acc_sum/self.test_cnt))
        f_log.write('\n')
        f_log.flush()
        self.test_acc_sum=0
        self.test_cnt=0
        np.save(f'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/{args.model}_label_{seed}',self.test_res)
        self.test_res=np.array([])
        print("Succesfully saved!")
        f_log.write("Succesfully saved!\n")
        f_log.flush()
    '''
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
    parser.add_argument('--sizes', type=str, default='40-20')
    parser.add_argument('--in-memory', default=True) # Always have to be true in this code.
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ver', type=int, default=0) # Used in ensemble step.
    parser.add_argument('--label_disturb_p', type=float, default=0.0)
    # Must specify seed everytime.
    # Batchsize, N_source need to be precisely selected, but don't change it for now.

    # python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.1 --batch_size=512
    # python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.2 --batch_size=512
    # python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.0 --batch_size=512
    # python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.1 --batch_size=256
    # python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.1 --batch_size=1024
    # python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.0 --batch_size=1024

    # python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.1 --batch_size=512 --ckpt=/users/PAS1289/oiocha/logs/baidu-NS_p=0.1_batch=512/lightning_logs/version_12988481/checkpoints/epoch=10-step=23902.ckpt
    # python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.1 --batch_size=512 --ckpt=/users/PAS1289/oiocha/logs/baidu-NS_p=0.1_batch=512/lightning_logs/version_12997170/checkpoints/epoch=2-step=6518.ckpt

    #CHINA-DEBUG
    #python OGB-NeurIPS-Team-Park/baidu_NS.py --label_disturb_p=0.0 --batch_size=512

    t0=time.time()
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)


    seed_everything(42)


    # Initialize log directory
    NROOT='/fs/scratch/PAS1289/data' # log file's root.
    path_log = NROOT+f'/baidu-NS_p={args.label_disturb_p}_batch={args.batch_size}.txt'
    f_log=open(path_log,'a')
    if args.ckpt!=None:
        f_log.write(f"Continue from...{args.ckpt}\n")
        f_log.flush()


    # Dataloader
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory, args.label_disturb_p)


    # Training
    if not args.evaluate:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        print("Device :",device)
        model = RGNN(args.model, datamodule.num_features+153,
                    datamodule.num_classes, args.hidden_channels,
                    datamodule.num_relations, num_layers=len(args.sizes),
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
                          default_root_dir=f'logs/baidu-NS_p={args.label_disturb_p}_batch={args.batch_size}',
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
        evaluator.save_test_submission(res, f'results/baidu-NS_p={args.label_disturb_p}_batch={args.batch_size}',mode='test-dev')
