from typing import Optional, Union, Dict

import os
import shutil
import os.path as osp

import torch
import numpy as np
from torch_sparse import SparseTensor
from matplotlib import pyplot as plt

"""
ROOT = '/tmp/slurmtmp.13184342/mag240m_kddcup2021'
__meta__ = torch.load(osp.join(ROOT, 'meta.pt'))
__split__ = torch.load(osp.join(ROOT, 'split_dict.pt'))
__rels__ = {
    ('author', 'paper'): 'writes',
    ('author', 'institution'): 'affiliated_with',
    ('paper', 'paper'): 'cites',
}

def edge_index(id1: str, id2: str, id3: Optional[str] = None) -> np.ndarray:
    src = id1
    rel, dst = (id3, id2) if id3 is None else (id2, id3)
    rel = __rels__[(src, dst)] if rel is None else rel
    name = f'{src}___{rel}___{dst}'
    path = osp.join(ROOT, 'processed', name, 'edge_index.npy')
    return np.load(path)


def get_idx_split(split: Optional[str] = None) -> Union[Dict[str, np.ndarray], np.ndarray]:
    return __split__ if split is None else __split__[split]


awp = torch.from_numpy(edge_index('author','writes','paper'))

a, p = awp
_, pc = torch.unique(p,return_counts=True)

np.save('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/mingi/pdist',pc.numpy())
plt.hist(pc.numpy(),bins=100)
print(len(pc))
plt.savefig('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/mingi/pdist')

_, ac = torch.unique(a,return_counts=True)
plt.clf()
np.save('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/mingi/adist',ac.numpy())
plt.hist(ac.numpy(),bins=100)
print(len(ac))
plt.savefig('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/mingi/adist')



spt = SparseTensor(row = p,col = a)
rowptr, col, val = spt.csr()

for i in range(100):
    print(rowptr[i])
"""

"""
pc = np.load('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/mingi/pdist.npy')
ac = np.load('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/mingi/adist.npy')

pcv, pcc = torch.unique(torch.from_numpy(pc),return_counts=True)
acv, acc = torch.unique(torch.from_numpy(ac),return_counts=True)

accs = torch.sum(acc).item()
pccs = torch.sum(pcc).item()

pcc = (pcc/pccs).numpy()
acc = (acc/accs).numpy()

plt.plot(pcc)
plt.savefig('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/mingi/pdist')

plt.clf()
plt.plot(acc)
plt.savefig('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/mingi/adist')

for i in range(100):
    print(pcv[i], pcc[i], acv[i], acc[i])
"""
ROOT = '/fs/ess/PAS1289/mag240m_kddcup2021'
__split__ = torch.load(osp.join(ROOT, 'split_dict.pt'))

def get_idx_split(split: Optional[str] = None) -> Union[Dict[str, np.ndarray], np.ndarray]:
    return __split__ if split is None else __split__[split]

path = osp.join(ROOT, 'processed', 'paper', 'node_label.npy')
label =  np.load(path)

train_idx = get_idx_split('train')
valid_idx = get_idx_split('valid')
test_idx = get_idx_split('test')

print(len(label), len(train_idx), len(valid_idx),len(test_idx))
print(label[train_idx[0]])