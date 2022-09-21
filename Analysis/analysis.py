
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
'''
tuple_list.sort()
counts.sort()
plt.plot(counts[::-100],'ro',markersize=2)
plt.savefig("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/Total_degree_dist.png")
plt.clf()
print(f"1st degree is {int(tuple_list[-1][1])} : {int(tuple_list[-1][0])}")
print(f"2nd degree is {int(tuple_list[-2][1])} : {int(tuple_list[-2][0])}")
print(f"3rd degree is {int(tuple_list[-3][1])} : {int(tuple_list[-3][0])}")
print("...")
print(f"Minimum degree is {int(tuple_list[0][1])} : {int(tuple_list[0][0])}")
print(f"Done! {time.time()-t0}s")
print()

print("Train degree distribution")
unique, counts = np.unique(row[train_idx], return_counts=True)
print(f"Num_zeros : {train_idx.shape[0]-len(unique)}")
print(f"Num_valid : {len(unique)}")
tuple_list=[]
for i in range(len(unique)):
    tuple_list.append((counts[i],unique[i]))
tuple_list.sort()
counts.sort()
plt.plot(counts[::-10],'ro',markersize=2)
plt.savefig("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/Train_degree_dist.png")
plt.clf()
print(f"1st degree is {int(tuple_list[-1][1])} : {int(tuple_list[-1][0])}")
print(f"2nd degree is {int(tuple_list[-2][1])} : {int(tuple_list[-2][0])}")
print(f"3rd degree is {int(tuple_list[-3][1])} : {int(tuple_list[-3][0])}")
print("...")
print(f"Minimum degree is {int(tuple_list[0][1])} : {int(tuple_list[0][0])}")
print(f"Done! {time.time()-t0}s")
print()

print("Valid degree distribution")
unique, counts = np.unique(row[valid_idx], return_counts=True)
print(f"Num_zeros : {valid_idx.shape[0]-len(unique)}")
print(f"Num_valid : {len(unique)}")
tuple_list=[]
for i in range(len(unique)):
    tuple_list.append((counts[i],unique[i]))
tuple_list.sort()
counts.sort()
plt.plot(counts[::-10],'ro',markersize=2)
plt.savefig("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/Valid_degree_dist.png")
plt.clf()
print(f"1st degree is {int(tuple_list[-1][1])} : {int(tuple_list[-1][0])}")
print(f"2nd degree is {int(tuple_list[-2][1])} : {int(tuple_list[-2][0])}")
print(f"3rd degree is {int(tuple_list[-3][1])} : {int(tuple_list[-3][0])}")
print("...")
print(f"Minimum degree is {int(tuple_list[0][1])} : {int(tuple_list[0][0])}")
print(f"Done! {time.time()-t0}s")
print()

print("Test-dev degree distribution")
unique, counts = np.unique(row[test_idx], return_counts=True)
print(f"Num_zeros : {test_idx.shape[0]-len(unique)}")
print(f"Num_valid : {len(unique)}")
tuple_list=[]
for i in range(len(unique)):
    tuple_list.append((counts[i],unique[i]))
tuple_list.sort()
counts.sort()
plt.plot(counts[::-10],'ro',markersize=2)
plt.savefig("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/Test_dev_degree_dist.png")
plt.clf()
print(f"1st degree is {int(tuple_list[-1][1])} : {int(tuple_list[-1][0])}")
print(f"2nd degree is {int(tuple_list[-2][1])} : {int(tuple_list[-2][0])}")
print(f"3rd degree is {int(tuple_list[-3][1])} : {int(tuple_list[-3][0])}")
print("...")
print(f"Minimum degree is {int(tuple_list[0][1])} : {int(tuple_list[0][0])}")
print(f"Done! {time.time()-t0}s")
print()

print("Test-challenge degree distribution")
unique, counts = np.unique(row[test_challenge_idx], return_counts=True)
print(f"Num_zeros : {test_challenge_idx.shape[0]-len(unique)}")
print(f"Num_valid : {len(unique)}")
tuple_list=[]
for i in range(len(unique)):
    tuple_list.append((counts[i],unique[i]))
tuple_list.sort()
counts.sort()
plt.plot(counts[::-10],'ro',markersize=2)
plt.savefig("/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/Test_challenge_degree_dist.png")
plt.clf()
print(f"1st degree is {int(tuple_list[-1][1])} : {int(tuple_list[-1][0])}")
print(f"2nd degree is {int(tuple_list[-2][1])} : {int(tuple_list[-2][0])}")
print(f"3rd degree is {int(tuple_list[-3][1])} : {int(tuple_list[-3][0])}")
print("...")
print(f"Minimum degree is {int(tuple_list[0][1])} : {int(tuple_list[0][0])}")
print(f"Done! {time.time()-t0}s")
#print("=======================================")
'''



'''
Loading done! : 0.13680410385131836s
num papers : 121751666
num authors : 122383112
num institution : 25721
num paper features : 768
num classes : 153
Train label distribution
Zero-cnt classes : 1
1st class is 100 : 79386
2nd class is 34 : 61031
3rd class is 14 : 58425
...
Minimum class is 99 : 72
Valid label distribution
Zero-cnt classes : 4
1st class is 140 : 9126
2nd class is 0 : 8474
3rd class is 141 : 4418
...
Minimum class is 96 : 16
=======================================
Loading symmetric datas...
Loading symmetric datas...
Row, col's length : 2593241212
Done! 21.15077304840088s

Total degree distribution
Num_zeros : 49243005
Num_valid : 72508661
1st degree is 89226997 : 242655
2nd degree is 55118718 : 190338
3rd degree is 84092676 : 184460
...
Minimum degree is 17 : 1
Done! 166.27183198928833s

Train degree distribution
Num_zeros : 701319
Num_valid : 411073
1st degree is 17107 : 1326
2nd degree is 1346767 : 1137
3rd degree is 5302615 : 991
...
Minimum degree is 7 : 1
Done! 193.27420616149902s

Valid degree distribution
Num_zeros : 61435
Num_valid : 77514
1st degree is 17107 : 275
2nd degree is 1346767 : 220
3rd degree is 1980718 : 162
...
Minimum degree is 415 : 1
Done! 193.53496408462524s

Test-dev degree distribution
Num_zeros : 27276
Num_valid : 60816
1st degree is 17107 : 76
2nd degree is 2854768 : 72
3rd degree is 1346767 : 71
...
Minimum degree is 3 : 1
Done! 193.68312120437622s

Test-challenge degree distribution
Num_zeros : 14652
Num_valid : 44074
1st degree is 17107 : 50
2nd degree is 2854768 : 48
3rd degree is 1346767 : 48
...
Minimum degree is 478 : 1
Done! 193.80322670936584s
'''