from typing import List, Optional, Tuple, NamedTuple, Union, Callable, Dict

import torch
from torch import Tensor
from torch_sparse import SparseTensor

import time
import ctypes


def sample(src: SparseTensor, num_neighbors: int,
           subset: Optional[torch.Tensor] = None) -> torch.Tensor:

    rowptr, col, _ = src.csr()
    rowcount = src.storage.rowcount()

    if subset is not None:
        rowcount = rowcount[subset]
        rowptr = rowptr[subset]
    else:
        rowptr = rowptr[:-1]

    rand = torch.rand((rowcount.size(0), num_neighbors), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.view(-1, 1))

    return col[rand]



def python_sample(rowptr: torch.Tensor, col: torch.Tensor, subset: torch.Tensor, num_neighbors: int, 
                replace: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass



def sample_adj(src: SparseTensor, subset: torch.Tensor, num_ppr_neighbors: int,
                num_atr_neighbors: int, num_ins_neighbors: int, replace: bool = False,
                relation_ptr: torch.Tensor=None) -> Tuple[SparseTensor, torch.Tensor]:

    rowptr, col, value = src.csr()

    rowptr, col, n_id, e_id = torch.ops.torch_sparse.sample_adj(
        rowptr, col, subset, num_ppr_neighbors, num_atr_neighbors, num_ins_neighbors, replace, relation_ptr)

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
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class NeighborSampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, relation_ptr: Optional[Tensor] = None, **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for Pytorch Lightning...
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes
        self.relation_ptr=relation_ptr

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                # print(f"DEBUG NS - value : {value}")
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(NeighborSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs = []
        n_id = batch
        for size in self.sizes:
            num_ppr, num_atr, num_ins = size
            # Implement above
            adj_t, n_id = sample_adj(self.adj_t, n_id, num_ppr, num_atr, num_ins, 
                                    replace=False, relation_ptr=self.relation_ptr)
            
            # retrieve original value after sample process
            e_id = adj_t.storage.value() 
            # Value(e_id) is used as idx to access original SparseTensor, 
            # since adj_t has zero-based new idx.
            size = adj_t.sparse_sizes()[::-1] # Transpose of size
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out, batch) if self.transform is not None else out
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
