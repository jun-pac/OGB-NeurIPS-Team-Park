
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import os
import os.path as osp
import sys
sys.path.insert(0,'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park')
import torch
from torch_sparse import SparseTensor
from ogb.lsc import MAG240MDataset

t0=time.time()

ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)
node_feat_path=osp.join(ROOT,'processed','paper','node_feat.npy')
node_label_path=osp.join(ROOT,'processed','paper','node_label.npy')
node_year_path=osp.join(ROOT,'processed','paper','node_year.npy')

print("h1",time.time()-t0)

print("num papers :",dataset.num_papers) # number of paper nodes
print("num authors :",dataset.num_authors) # number of author nodes
print("num institution :",dataset.num_institutions) # number of institution nodes
print("num paper features :",dataset.num_paper_features) # dimensionality of paper features
print("num classes :",dataset.num_classes) # number of subject area classes

sample_sz=40000
np.random.seed(42)


'''
train_idx=dataset.get_idx_split('train') # (1112392, 768). Loading all feature takes 80s. 
origin_idx=np.random.randint(0,1112392,sample_sz)
sample_idx=train_idx[origin_idx]
#paper_feat=dataset.paper_feat[train_idx] # this should be run on normal node.
#paper_label=dataset.paper_label[train_idx]
#paper_year=dataset.paper_year[train_idx]
paper_feat=dataset.paper_feat[sample_idx]
paper_label=dataset.paper_label[sample_idx]
paper_year=dataset.paper_year[sample_idx]

model=TSNE(n_components=2, verbose=1, random_state=123, learning_rate=0.01, n_iter=25000) # previously 0.01
# This takes 180s for 2000 points
# 1000s for 20000 points
tsne_pos=model.fit_transform(paper_feat)
t_pos=np.transpose(tsne_pos)
print(type(tsne_pos)) # tsne_pos : (sample_sz,2) shape ndarray 
print(tsne_pos.shape)
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/tsne_pos"+str(sample_sz),tsne_pos)
np.save("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/tsne_label"+str(sample_sz),paper_label)
print("h2",time.time()-t0) 
'''
val_idx=dataset.get_idx_split('valid')
test_idx=dataset.get_idx_split('test-dev')
test_challenge_idx=dataset.get_idx_split('test-challenge')
print("val :",val_idx.shape[0]) 
print("test-dev :",test_idx.shape[0])
print("test-challenge :",test_challenge_idx.shape[0])

t_pos=np.transpose(np.load("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/tsne_pos"+str(sample_sz)+".npy"))
paper_label=np.load("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/tsne_label"+str(sample_sz)+".npy")

for i in range(10,25): #[5,10,15,20,25,30,40,50,60,70,80,90]: #dataset.num_classes
    idx=np.where(paper_label==i)
    plt.scatter(t_pos[0][idx],t_pos[1][idx],s=2)
print("h3",time.time()-t0)
plt.savefig(osp.join("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park","TSNE-"+str(sample_sz)+".png"))

'''
num papers : 121751666
num authors : 122383112
num institution : 25721
num paper features : 768
num classes : 153
train : 1112392
val : 138949
test-dev : 88092
test-challenge : 58726
'''