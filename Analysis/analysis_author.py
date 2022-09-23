
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
np.random.seed(42)

print(f"Loading done! : {time.time()-t0}s")

print("num papers :",dataset.num_papers) # number of paper nodes
print("num authors :",dataset.num_authors) # number of author nodes
print("num institution :",dataset.num_institutions) # number of institution nodes
print("num paper features :",dataset.num_paper_features) # dimensionality of paper features
print("num classes :",dataset.num_classes) # number of subject area classes
print()

# Distribution of label
print("Train label distribution")
train_idx=dataset.get_idx_split('train')
valid_idx=dataset.get_idx_split('valid')
test_idx=dataset.get_idx_split('test-dev')
test_challenge_idx=dataset.get_idx_split('test-challenge')

train_label=dataset.paper_label[train_idx]
unique, counts = np.unique(train_label, return_counts=True)
tuple_list=[]
for i in range(len(unique)):
    tuple_list.append((counts[i],unique[i]))
tuple_list.sort()
none_cnt=153-len(unique)
zero_cnt=0
for i in range(len(unique)):
    if tuple_list[i][1]==0.:
        zero_cnt+=1
print(f"Zero-cnt classes : {zero_cnt+none_cnt}")
print(f"1st class is {int(tuple_list[-1][1])} : {int(tuple_list[-1][0])}")
print(f"2nd class is {int(tuple_list[-2][1])} : {int(tuple_list[-2][0])}")
print(f"3rd class is {int(tuple_list[-3][1])} : {int(tuple_list[-3][0])}")
print("...")
print(f"Minimum class is {int(tuple_list[zero_cnt][1])} : {int(tuple_list[zero_cnt][0])}")
plt.plot(unique, counts, 'o', markersize=2)
plt.savefig("Train_label_dist.png")
plt.clf()
print()

print("Valid label distribution")
valid_label=dataset.paper_label[valid_idx]
unique, counts = np.unique(valid_label, return_counts=True)
tuple_list=[]
for i in range(len(unique)):
    tuple_list.append((counts[i],unique[i]))
tuple_list.sort()
none_cnt=153-len(unique)
zero_cnt=0
for i in range(len(unique)):
    if tuple_list[i][1]==0.:
        zero_cnt+=1
print(f"Zero-cnt classes : {zero_cnt+none_cnt}")
print(f"1st class is {int(tuple_list[-1][1])} : {int(tuple_list[-1][0])}")
print(f"2nd class is {int(tuple_list[-2][1])} : {int(tuple_list[-2][0])}")
print(f"3rd class is {int(tuple_list[-3][1])} : {int(tuple_list[-3][0])}")
print("...")
print(f"Minimum class is {int(tuple_list[zero_cnt][1])} : {int(tuple_list[zero_cnt][0])}")
plt.plot(unique, counts, 'o', markersize=2)
plt.savefig("Valid_label_dist.png")
plt.clf()
print()
print("=======================================")
print()

# Now, degree distribution
t0=time.time()
print("Loading symmetric datas...")
Adj=torch.load('/fs/ess/PAS1289/mag240m_kddcup2021/paper_to_paper_symmetric.pt').coo()
row,col,val=Adj
print(f"Row, col's length : {row.shape[0]}")
print(f"Done! {time.time()-t0}s")
print()
# You can actually sort whole array with code such as #row=row[:,np.argsort(row)] #col=col[:,np.argsort(row)] 
# But that's unnecessary for now

degree=np.zeros(dataset.num_papers)

print("Total degree distribution")
unique, counts = np.unique(row, return_counts=True)
print(f"Num_zeros : {dataset.num_papers-len(unique)}")
print(f"Num_valid : {len(unique)}")
tuple_list=[]
for i in range(len(unique)):
    tuple_list.append((counts[i],unique[i]))
    degree[unique[i]]=counts[i]
np.save("paper_degree",degree)
