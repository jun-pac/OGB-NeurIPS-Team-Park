from typing import List, Optional, Tuple, NamedTuple, Union, Callable, Dict

import torch
from torch import Tensor
from torch_sparse import SparseTensor
import numpy as np

import time
import ctypes


def python_sample(rowptr: torch.Tensor, col: torch.Tensor, idx: torch.Tensor,num_ppr_neighbors: int,
                num_atr_neighbors: int, num_ins_neighbors: int, relation_ptr: torch.Tensor=None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # replace is always true.
    out_rowptr = torch.empty(idx.numel() + 1, dtype=rowptr.dtype) # rowptr.options()
    out_rowptr[0]=0
    cols=[]
    n_ids=[]
    n_id_map={}

    #print(f"INPUTS : {rowptr, col, num_ppr_neighbors, num_atr_neighbors, num_ins_neighbors, relation_ptr}")
    #print(f"idx dtype : {idx.dtype}")
    for n in range(idx.numel()):
        i = int(idx[n])
        cols.append([])
        n_id_map[i] = n
        n_ids.append(i)

    # Sample begin (always without replacement)
    for i in range(idx.numel()):
        n = int(idx[i])
        row_start = int(rowptr[n])
        ppr_count=int(relation_ptr[3*n+1]-relation_ptr[3*n])
        atr_count=int(relation_ptr[3*n+2]-relation_ptr[3*n+1])
        ins_count=int(relation_ptr[3*n+3]-relation_ptr[3*n+2])
        perm=set()

        # Sample ppr
        if(ppr_count <= num_ppr_neighbors):
            for j in range(ppr_count):
                perm.add(j)
        else :
            for j in range(ppr_count-num_ppr_neighbors, ppr_count):
                temp=np.random.randint(0,j)
                if (not temp in perm):
                    perm.add(temp)
                else:
                    perm.add(j)

        # Sample atr
        if(atr_count <= num_atr_neighbors):
            for j in range(atr_count):
                perm.add(ppr_count+j)
        else :
            for j in range(atr_count-num_atr_neighbors, atr_count):
                temp=np.random.randint(0,j)
                if (not temp in perm):
                    perm.add(ppr_count+temp)
                else:
                    perm.add(ppr_count+j)

        # Sample ins
        if(ins_count <= num_ins_neighbors):
            for j in range(ins_count):
                perm.add(ppr_count+atr_count+j)
        else :
            for j in range(ins_count-num_ins_neighbors, ins_count):
                temp=np.random.randint(0,j)
                if (not temp in perm):
                    perm.add(ppr_count+atr_count+temp)
                else:
                    perm.add(ppr_count+atr_count+j)

        for p in perm:
            e = int(row_start + p)
            c = int(col[e]) # As this is csr format, c is real idx of node.

            if (not c in n_id_map):
                n_id_map[c] = len(n_ids)
                # n_id_map is unordered_map : n_idx's value -> n_idx's internal idx
                # don't increase its size 
                n_ids.append(c)

            cols[i].append((n_id_map[c], e))
            # cols : vector<vector<tuple<int,int>>>, first dimension has len(n_idx) size
            # Store col, e_id information
            # n_id_map[c] is sampled node's pseudo idx, e is sampled node's real idx (order of selected edge in TOTAL edges.) 
            # I just have understood that e_id is real EDGE index, NOT real NODE index.

        out_rowptr[i + 1] = out_rowptr[i] + len(cols[i])
        # Generating new rowptr.
        # Of course it interact with pseudo idx.
        # I think out_rowptr is ptr of out_rowptr
    N = len(n_ids)
    # out_n_id = torch.from_blob(n_ids.data(), {N}, col.options()).clone()
    # out_n_id=torch.clone(torch.Tensor(n_ids).to(dtype=col.dtype))
    out_n_id=torch.Tensor(N).to(dtype=torch.int64)
    for i in range(N):
        out_n_id[i]=n_ids[i]

    E = out_rowptr[idx.numel()] # Total size of sampled edges. 
    out_col = torch.empty(E, dtype=torch.int64) # col.options()
    out_e_id = torch.empty(E, dtype=torch.int64) # col.options()

    i = 0
    for col_vec in cols:
        col_vec.sort(key=lambda x:x[0])
        for value in col_vec:
            out_col[i] = value[0] # New node ordering
            out_e_id[i] = value[1] # original edge ordering
            i += 1

    #print(f"OUTPUTS : {out_rowptr, out_col, out_n_id, out_e_id}")
    return (out_rowptr, out_col, out_n_id, out_e_id)



def sample_adj(src: SparseTensor, subset: torch.Tensor, num_ppr_neighbors: int,
                num_atr_neighbors: int, num_ins_neighbors: int,
                relation_ptr: torch.Tensor=None) -> Tuple[SparseTensor, torch.Tensor]:

    rowptr, col, value = src.csr()
    
    '''
    # Use when cpp dll complete
    rowptr, col, n_id, e_id = torch.ops.torch_sparse.sample_adj(
        rowptr, col, subset, num_ppr_neighbors, num_atr_neighbors, num_ins_neighbors, replace, relation_ptr)
    '''
    rowptr, col, n_id, e_id = python_sample(
        rowptr, col, subset, num_ppr_neighbors, num_atr_neighbors, num_ins_neighbors, relation_ptr)
    
    if value is not None:
        value = value[e_id]
    # value will be replaced

    out = SparseTensor(rowptr=rowptr, row=None, col=col, value=value,
                       sparse_sizes=(subset.size(0), n_id.size(0)),
                       is_sorted=True)

    return out, n_id

class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        return Adj(adj_t, self.size)

# How to check whether we sampled valid edges?
class NeighborSampler(torch.utils.data.DataLoader):
    def __init__(self, mono_edge_index: Union[Tensor, SparseTensor],
                 edge_index: Union[Tensor, SparseTensor],
                 sizes: List[List[int]], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, mono_relation_ptr:  Optional[Tensor] = None,
                 relation_ptr: Optional[Tensor] = None, **kwargs):

        mono_edge_index = mono_edge_index.to('cpu')
        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for Pytorch Lightning...
        self.mono_edge_index = mono_edge_index
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes
        self.mono_relation_ptr=mono_relation_ptr
        self.relation_ptr=relation_ptr

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.mono__val__ = None
        self.__val__ = None

        mono_adj_t = mono_edge_index
        adj_t = edge_index
        if return_e_id:
            self.mono__val__ = mono_adj_t.storage.value()
            self.__val__ = adj_t.storage.value()
            mono_value = torch.arange(mono_adj_t.nnz())
            value = torch.arange(adj_t.nnz())
            mono_adj_t = mono_adj_t.set_value(mono_value, layout='coo')
            adj_t = adj_t.set_value(value, layout='coo')
        self.mono_adj_t = mono_adj_t
        self.adj_t = adj_t

        super(NeighborSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        #print("Sample begin : ")
        t0=time.time()
        batch_size: int = len(batch)
        adjs = adjs = [0 for _ in range(len(self.sizes)*2)]
        n_id = batch

        for i, size in enumerate(self.sizes):
            num_ppr, num_atr, num_ins = size
            cur_adj_t = self.mono_adj_t if i==0 else self.adj_t
            cur_relation_ptr = self.mono_relation_ptr if i==0 else self.relation_ptr
            cur__val__= self.mono__val__ if i==0 else self.__val__

            cur_adj_t, n_id = sample_adj(cur_adj_t, n_id, num_ppr, num_atr, num_ins, relation_ptr=cur_relation_ptr)
            e_id = cur_adj_t.storage.value() 
            size = cur_adj_t.sparse_sizes()[::-1] # Transpose of size
            if cur__val__ is not None:
                cur_adj_t.set_value_(cur__val__[e_id], layout='coo')

            row, col, val = cur_adj_t.coo()

            cur_adj_t_t = SparseTensor(row = col, col = row, value = val)
            size_t = cur_adj_t.sparse_sizes()


            adjs[-1-i] = Adj(cur_adj_t, (0,0))
            adjs[i] = Adj(cur_adj_t_t, (1,1))

        #adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)        
        out = self.transform(*out) if self.transform is not None else out
        #print(f"Sample Time : {time.time()-t0}")
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
