# Bidirectional full

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
                 in_memory: bool = False, label_disturb_p: float = 0,
                 time_disturb_p: float = 0.2, bit: int=10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory
        self.label_disturb_p=label_disturb_p
        self.time_disturb_p=time_disturb_p
        self.bit=bit
        self.visit_flag=visit_flag

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
        self.train_idx.share_memory_()
        self.train_label=dataset.paper_label[self.train_idx]
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx.share_memory_()
        self.test_challenge_idx = torch.from_numpy(dataset.get_idx_split('test-challenge'))
        self.test_challenge_idx.share_memory_()

        if args.embedding:
            print(f"Load visit flag...")
            t_=time.time()
            if os.path.isfile(visit_idx_path):
                self.visit_idx=torch.load(visit_idx_path)
            else:
                self.loaded_visit_flag = torch.load(visit_path)
                self.visit_idx = torch.squeeze(torch.nonzero(self.loaded_visit_flag))
                torch.save(self.visit_idx, visit_idx_path)
            # Continue from last file
            
            global embedding_number
            global embedding_file_number
            embedding_file_number=args.embedding_file_start
            embedding_number = len(self.visit_idx) - 1280000*args.embedding_file_start
            for i in range(42539880//1280000+1):
                if not os.path.isfile(embedding_path+str(embedding_file_number)+'.pt'):
                    break
                embedding_file_number+=1
                if i != 42539880//1280000:
                    embedding_number-=1280000
                else:
                    embedding_number-=42539880%1280000
            assert embedding_number>0
            self.visit_idx=self.visit_idx[-embedding_number:]
            print(f"Done! {time.time()-t_:.2f}, Start forward {embedding_file_number}th file, {embedding_number} nodes left.")

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        self.N=N 
        
        t1=time.time()
        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(N, self.num_features))
        self.x = np.empty((N, self.num_features), dtype=np.float16)
        if args.debug:
            print("Mingi debug activate")
            self.x = torch.from_numpy(self.x)
        else:
            self.x[:] = x
            self.x = torch.from_numpy(self.x).share_memory_()
        print(f"full_feat loading time : {time.time()-t1:.2f}")
        
        if not args.embedding:
            self.y = torch.from_numpy(dataset.all_paper_label)
        else :
            self.y=torch.from_numpy(np.empty(N))
            print(f"self.y.shape : {self.y.shape}")

        path='/fs/ess/PAS1289/mag240m_kddcup2021/full_adj_t.pt'
        self.adj_t = torch.load(path)
        one_hot_encoding = np.eye(153)[[int(x) for x in self.train_label]].astype(np.float16)
        self.one_hot_dict={}
        for i in range(len(self.train_idx)):
            self.one_hot_dict[int(self.train_idx[i])]=one_hot_encoding[i]

        self.relation_ptr=torch.load("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/sampler/relation_ptr.pt")
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
        self.paper_year=dataset.paper_year # load years to memory!
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

    def test_dataloader(self):  
        node_idx=self.val_idx if not args.embedding else self.visit_idx
        return NeighborSampler(self.adj_t, node_idx=node_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=10, relation_ptr=self.relation_ptr)

    def hidden_test_dataloader(self): # This is test-dev. Not test-challenge
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=10, relation_ptr=self.relation_ptr)

    def test_visit_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=torch.cat((self.test_idx, self.test_challenge_idx), dim=0),
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch_test,
                               batch_size=self.batch_size, num_workers=5, relation_ptr=self.relation_ptr)

    def convert_batch_train(self, batch_size, n_id, adjs):
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

        if args.visit_check:
            visit_flag[n_id]=1
        
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

        # sejun debug
        if(args.debug):
            adj_t,_a,_b=adjs[0]
            row,col,_=adj_t.coo()
            cnt1=0
            cnt2=0
            for i in range(len(row)):
                if(n_id[row[i]]>=self.num_papers or n_id[col[i]]>=self.num_papers):
                    continue
                if(self.paper_year[n_id[row[i]]] < self.paper_year[n_id[col[i]]]):
                    cnt1+=1
                else:
                    cnt2+=1
            print(f"2nd sample | row<col : {cnt1}, row>=col : {cnt2} | sample size : {adj_t.sizes()}")
            f_log.write(f"2nd sample | row<col : {cnt1}, row>=col : {cnt2} | sample size : {adj_t.sizes()}\n")
            f_log.flush()

            adj_t,_a,_b=adjs[1]
            row,col,_=adj_t.coo()
            cnt1=0
            cnt2=0
            for i in range(len(row)):
                if(n_id[row[i]]>=self.num_papers or n_id[col[i]]>=self.num_papers):
                    continue
                if(self.paper_year[n_id[row[i]]] < self.paper_year[n_id[col[i]]]):
                    cnt1+=1
                else:
                    cnt2+=1
            print(f"1st sample | row<col : {cnt1}, row>=col : {cnt2} | sample size : {adj_t.sizes()}")
            f_log.write(f"1st sample | row<col : {cnt1}, row>=col : {cnt2} | sample size : {adj_t.sizes()}\n")
            f_log.flush()

        if args.visit_check:
            if args.debug:
                print(f"CONV BATCH nonzero1 - {torch.count_nonzero(visit_flag)}, len(n_id) : {len(n_id)}")
            visit_flag[n_id]=1
            if args.debug:
                print(f"CONV BATCH nonzero2 - {torch.count_nonzero(visit_flag)}")
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
        self.test_num=0
        self.embedding_list=[]

    def save_embedding(self):
        global embedding_file_number
        print(f"{embedding_file_number}th result saving... {time.time()-t0}")
        f_log.write(f"{embedding_file_number}th result saving... {time.time()-t0}\n")
        f_log.flush()
        result=torch.cat(self.embedding_list, dim=0)
        if args.debug:
            print(f"shape of reduced embedding_list : {result.shape}")
            f_log.write(f"shape of reduced embedding_list : {result.shape}")
            f_log.flush()
        torch.save(result, embedding_path+str(embedding_file_number)+'.pt')
        f_log.flush()
        self.embedding_list=[]
        embedding_file_number+=1
        assert not os.path.isfile(embedding_path+str(embedding_file_number)+'.pt')

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
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
        if((args.debug and batch_idx%10==0) or batch_idx%100==0):
            print(f"{self.current_epoch} epoch ; {name[1:]} | train_acc : {tmp_acc:.5f} | time : {time.time()-t0:.2f} | batch : {batch_idx}/{1112392//args.batch_size}")
            f_log.write(f"{self.current_epoch} epoch ; {name[1:]} | train_acc : {tmp_acc:.5f} | time : {time.time()-t0:.2f} | batch : {batch_idx}/{1112392//args.batch_size}\n")
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
        if((args.debug and batch_idx%10==0) or batch_idx%50==0):
            print(f"{self.current_epoch} epoch ; {name[1:]} | valid_acc : {tmp_acc:.5f} | time : {time.time()-t0:.2f} | batch : {batch_idx}/{138949//args.batch_size}")
            f_log.write(f"{self.current_epoch} epoch ; {name[1:]} | valid_acc : {tmp_acc:.5f} | time : {time.time()-t0:.2f} | batch : {batch_idx}/{138949//args.batch_size}\n")
            f_log.flush()

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        if not args.embedding:
            tmp_acc=self.test_acc(y_hat.softmax(dim=-1), batch.y).item() # What is the type of this value?
            self.test_acc_sum+=batch.x.shape[0]*tmp_acc
            self.test_cnt+=batch.x.shape[0]
            self.test_res.append(y_hat.softmax(dim=-1).cpu().numpy())
            self.log('test_acc', tmp_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if((args.debug and batch_idx%10==0) or batch_idx%50==0):
                print(f"{name[1:]} | test_acc : {tmp_acc:.5f} | time : {time.time()-t0:.2f} | batch : {batch_idx}/{138949//128}")
                f_log.write(f"{name[1:]} | test_acc : {tmp_acc:.5f} | time : {time.time()-t0:.2f} | batch : {batch_idx}/{138949//128}\n")
                f_log.flush()
        else:
            self.embedding_list.append(y_hat)
            if((args.debug and batch_idx%10==0) or batch_idx%1000==999):
                print(f"{name[1:]} | time : {time.time()-t0:.2f} | batch : {batch_idx}/{embedding_number//128}")
                f_log.write(f"{name[1:]} | time : {time.time()-t0:.2f} | batch : {batch_idx}/{embedding_number//128}\n")
                f_log.flush()
            if((args.debug and batch_idx%10==0) or batch_idx%10000==9999):
                self.save_embedding()
    
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
            print("Successfully saved!")
            f_log.write("Successfully saved!\n")
        if args.visit_check:
            torch.save(visit_flag, visit_path)
            print(f"Visit flag saved, nonzero element : {torch.count_nonzero(visit_flag)}")
            f_log.write(f"Visit flag saved, nonzero element : {torch.count_nonzero(visit_flag)}\n")
        f_log.flush()
        self.val_res=[]
        self.val_acc_sum=0
        self.val_cnt=0
        self.batch_idx=0

    def test_epoch_end(self, outputs) -> None:
        if not args.embedding:
            print("Test Epoch end... Accuracy : "+str(self.test_acc_sum/self.test_cnt))
            f_log.write("Test Epoch end... Accuracy : "+str(self.test_acc_sum/self.test_cnt))
            f_log.write('\n')
            if self.test_acc_sum/self.test_cnt>self.max_val_acc:
                self.max_val_acc=self.test_acc_sum/self.test_cnt
                self.test_res=np.concatenate(self.test_res)
                np.save(f'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/test_activation'+name, self.test_res)
                print("Successfully saved!")
                f_log.write("Successfully saved!\n")
            f_log.flush()
            self.test_res=[]
            self.test_acc_sum=0
            self.test_cnt=0
        else:
            self.save_embedding()
            global embedding_file_number
            embedding_file_number=0

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
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--visit_check', action='store_true')
    parser.add_argument('--visit_test', action='store_true')
    parser.add_argument('--embedding_file_start', type=int, default=0)
    # Must specify seed everytime.
    # Batchsize, N_source need to be precisely selected, but don't change it for now.


    # DEBUG
    # python OGB-NeurIPS-Team-Park/bi_full.py --label_disturb_p=0.1 --batch_size=1024 --visit_check --visit_test --debug
    
    # MAIN
    # python OGB-NeurIPS-Team-Park/bi_full.py --label_disturb_p=0.1 --batch_size=1024 --visit_check --ckpt=/users/PAS1289/oiocha/logs/bi-full_p=0.1_batch=1024/lightning_logs/version_13185592/checkpoints/epoch=32-step=35198.ckpt

    # TEST
    # python OGB-NeurIPS-Team-Park/bi_full.py --evaluate --label_disturb_p=0.0 --time_disturb_p=0.0 --batch_size=1024 --ckpt=/users/PAS1289/oiocha/logs/bi-full_p=0.1_batch=1024/lightning_logs/version_13082877/checkpoints/epoch=34-step=38044.ckpt
    
    # Evaluate
    # python OGB-NeurIPS-Team-Park/bi_full.py --embedding --label_disturb_p=0.0 --time_disturb_p=0.0 --batch_size=1024 --evaluate --ckpt=/users/PAS1289/oiocha/logs/bi-full_p=0.1_batch=1024/lightning_logs/version_13082877/checkpoints/epoch=34-step=38044.ckpt
    
    # Embedding
    # python OGB-NeurIPS-Team-Park/bi_full.py --embedding --debug --ckpt=/users/PAS1289/oiocha/logs/bi-full_p=0.1_batch=1024/lightning_logs/version_13217996/checkpoints/epoch=42-step=46068.ckpt
    # python OGB-NeurIPS-Team-Park/bi_full.py --embedding --embedding_file_start=0 --ckpt=/users/PAS1289/oiocha/logs/bi-full_p=0.1_batch=1024/lightning_logs/version_13217996/checkpoints/epoch=42-step=46068.ckpt
    # python OGB-NeurIPS-Team-Park/bi_full.py --embedding --embedding_file_start=16 --ckpt=/users/PAS1289/oiocha/logs/bi-full_p=0.1_batch=1024/lightning_logs/version_13217996/checkpoints/epoch=42-step=46068.ckpt
    
    seed_everything(42)
    t0=time.time()
    args = parser.parse_args()
    #args.sizes = [int(i) for i in args.sizes.split('-')]
    sizes=[[40,10,0],[15,10,5]] # Default behavior
    print(args)
    print(f"Sizes : {sizes}")

    import pytorch_lightning as pl
    if args.debug:
        print(f"pytorch_lightning.__version__ : {pl.__version__}")


    # Initialize log directory
    if args.debug:
        name=f'/bi-full_DEBUG'
    elif args.ckpt!=None:
        name='/'+args.ckpt.split('/')[5]
    elif args.hidden_channels==1024:
        name=f'/bi-full_p={args.label_disturb_p}_batch={args.batch_size}'
    else:
        name=f'/bi-full_p={args.label_disturb_p}_batch={args.batch_size}_hidden={args.hidden_channels}'


    NROOT='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/txtlog' # log file's root.
    path_log = NROOT+name+'.txt'
    f_log=open(path_log,'a')
    if args.ckpt!=None:
        f_log.write(f"Continue from...{args.ckpt}\n")
        f_log.flush()

    # Visit checker
    # Solve test visit flag issue
    if args.visit_check or args.embedding:
        if args.debug:
            visit_path='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/visit_flag'+'/'+args.ckpt.split('/')[5]+'.pt'
        else:
            visit_path='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/visit_flag'+name+'.pt'
        visit_idx_path='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/visit_idx_flag'+name+'.pt'
        embedding_path='/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/embedding'+name
        if os.path.isfile(visit_path):
            visit_flag=torch.load(visit_path)
        else:
            visit_flag=torch.zeros(244160500) # 244160499 + 1
        visit_flag.share_memory_()
        embedding_number=0

    # Dataloader
    datamodule = MAG240M(ROOT, args.batch_size, sizes, args.in_memory, args.label_disturb_p, args.time_disturb_p, args.bit)

    # Visit testnode
    if args.visit_check and args.visit_test:
        datamodule.setup()
        print(f"Test visit flag update. Current non-zero entries : {torch.count_nonzero(visit_flag)}")
        f_log.write(f"Test visit flag update. Current non-zero entries : {np.count_nonzero(visit_flag)}\n")
        f_log.flush()
        testloader=datamodule.test_visit_dataloader()
        for i,batch in enumerate(testloader):
            if(i%10==0):
                print(f"Test visit flag {i}/{(88092+58726)//args.batch_size}th batch | time : {time.time()-t0:.2f} | Nonzero : {torch.count_nonzero(visit_flag)}")
                f_log.write(f"Test visit flag {i}/{(88092+58726)//args.batch_size}th batch | time : {time.time()-t0:.2f} | Nonzero : {np.count_nonzero(visit_flag)}\n")
                f_log.flush()
        print(f"Done! ... {time.time()-t0:.2f}")
        f_log.write(f"Done! ... {time.time()-t0:.2f}\n")
        
        torch.save(visit_flag,visit_path)
        print("Visit flag saved.")
        f_log.write("Visit flag saved.\n")
        f_log.flush()

    if args.debug:
        print(f"Visit flag nonzero : {torch.count_nonzero(visit_flag)}")

    # Training
    if not (args.evaluate or args.embedding):
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        print("Device :",device)
        
        if args.ckpt != None:
            checkpoint = torch.load(args.ckpt)
            model=RGNN.load_from_checkpoint(args.ckpt)

            if (args.debug):
                # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters'])
                print(f"checkpoint['epoch'] : {checkpoint['epoch']}")
                print(f"checkpoint['global_step'] : {checkpoint['global_step']}")
                print(f"checkpoint.keys() : {checkpoint.keys()}")
                #print(f"BEFORE model.optimizers() : {model.optimizers()}")

        else:
            model = RGNN(args.model, datamodule.num_features+153+args.bit,
                    datamodule.num_classes, args.hidden_channels,
                    datamodule.num_relations, num_layers=len(sizes),
                    dropout=args.dropout)

        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',save_top_k=3)
        # tensorboard --logdir=/users/PAS1289/oiocha/logs/rgat/lightning_logs
        if args.ckpt != None:
            trainer = Trainer(max_epochs=args.epochs,
                            callbacks=[checkpoint_callback],
                            default_root_dir='logs'+name,
                            progress_bar_refresh_rate=0, resume_from_checkpoint=args.ckpt)
        else:
            trainer = Trainer(max_epochs=args.epochs,
                            callbacks=[checkpoint_callback],
                            default_root_dir='logs'+name,
                            progress_bar_refresh_rate=0)
        if args.debug:
            print(f"AFTER model.optimizers() : {trainer.optimizers}")
            print(f"model.global_step : {model.global_step}")
            print(f"model.current_epoch : {model.current_epoch}")

        trainer.fit(model, datamodule=datamodule)
        
    if args.embedding:
        assert args.ckpt!=None
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        ckpt=args.ckpt
        trainer = Trainer(resume_from_checkpoint=ckpt,
                          progress_bar_refresh_rate=0) # gpus=args.device,
        model = RGNN.load_from_checkpoint(args.ckpt)

        datamodule.batch_size = 128 # initially 16
        datamodule.sizes = [[80,20,0],[30,20,10]] 
        trainer.test(model=model, datamodule=datamodule)
    
    # Evaluate
    elif args.evaluate:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        # Ignore previous code
        ckpt=args.ckpt
        trainer = Trainer(resume_from_checkpoint=ckpt,
                          progress_bar_refresh_rate=0) # gpus=args.device,
        
        model = RGNN.load_from_checkpoint(args.ckpt)

        datamodule.batch_size = 16*8 # initially 16
        
        f_log.write("Original : [[40,10,0],[20,10,5]]\n")
        f_log.flush()
        datamodule.sizes = [[40,10,0],[20,10,5]] 
        trainer.test(model=model, datamodule=datamodule)

        f_log.write("X3 : [[120,30,0],[60,30,15]]\n")
        f_log.flush()
        datamodule.sizes = [[120,30,0],[60,30,15]] 
        trainer.test(model=model, datamodule=datamodule)

        f_log.write("X4 : [[160,40,0],[80,40,20]]\n")
        f_log.flush()
        datamodule.sizes = [[160,40,0],[80,40,20]] 
        trainer.test(model=model, datamodule=datamodule)

        f_log.write("X5 : [[200,50,0],[100,60,25]]\n")
        f_log.flush()
        datamodule.sizes = [[200,50,0],[100,60,25]]  # (Almost) no sampling...
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