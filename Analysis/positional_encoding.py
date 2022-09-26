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
np.random.seed(42)
t0=time.time()

ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)

print("num papers :",dataset.num_papers) # number of paper nodes
print("num authors :",dataset.num_authors) # number of author nodes
print("num institution :",dataset.num_institutions) # number of institution nodes
print("num paper features :",dataset.num_paper_features) # dimensionality of paper features
print("num classes :",dataset.num_classes) # number of subject area classes
print()

train_idx=dataset.get_idx_split('train')
valid_idx=dataset.get_idx_split('valid')
test_idx=dataset.get_idx_split('test-dev')
test_challenge_idx=dataset.get_idx_split('test-challenge')

train_year=dataset.paper_year[train_idx]
valid_year=dataset.paper_year[valid_idx]
test_year=dataset.paper_year[test_idx]
test_challenge_year=dataset.paper_year[test_challenge_idx]
year_m1=dataset.paper_year
year_m=dataset.paper_year[:]

t1=time.time()
for i in range(0,100000000,10000):
    a=dataset.paper_year[i]
print(f"Disk random access 10000 : {time.time()-t1}")

t1=time.time()
for i in range(0,100000000,10000):
    a=year_m1[i]
print(f"Memory random access 10000 : {time.time()-t1}")
print()

print(f"Train          : {train_year[:10]}, Max : {max(train_year)}, Min : {min(train_year)}")
print(f"Valid          : {valid_year[:10]}, Max : {max(valid_year)}, Min : {min(valid_year)}")
print(f"Test           : {test_year[:10]}, Max : {max(test_year)}, Min : {min(test_year)}")
print(f"Test-challenge : {test_challenge_year[:10]}, Max : {max(test_challenge_year)}, Min : {min(test_challenge_year)}")

cnt_2021=0
for i in test_year:
    if i==2021:
        cnt_2021+=1
print(f"Test 2021 : {cnt_2021}, 2020 : {len(test_year)-cnt_2021}")

cnt_2021=0
for i in test_challenge_year:
    if i==2021:
        cnt_2021+=1
print(f"Test_challenge 2021 : {cnt_2021}, 2020 : {len(test_challenge_year)-cnt_2021}")

'''
Train          : [2014 2014 2015 2005 2013 2003 2015 2016 2015 2015], Max : 2018, Min : 1941
Valid          : [2019 2019 2019 2019 2019 2019 2019 2019 2019 2019], Max : 2019, Min : 2019
Test           : [2020 2020 2020 2020 2020 2020 2020 2020 2020 2020], Max : 2021, Min : 2020
Test-challenge : [2020 2020 2020 2020 2020 2020 2020 2020 2020 2020], Max : 2021, Min : 2020
'''

# Build positional encoding
_idx=[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,4,4,5,5,6,7,8,9,9]
year_to_idx=[0]*1981+_idx
bit=10
positional_encoding=[]
for i in range(bit):
    wave=np.arange(bit)
    wave=np.cos((wave-i)*np.pi/10)
    pos_row=torch.from_numpy(wave)
    #print(f"{i}th encoding : {pos_row}")
    positional_encoding.append(pos_row)


'''
print(f"type(train_idx) : {type(train_idx)}")
print(f"type(train_year) : {type(train_year)}")
print(f"type(dataset.paper_year) : {type(dataset.paper_year)}")
print(f"dataset.paper_year.shape : {dataset.paper_year.shape}")
print(f"type(year_m) : {type(year_m)}")
print(f"year_m[range(0,100000000,10000000)] : {year_m[range(0,100000000,10000000)]}")

for i in range(10):
    print(int(4+torch.randn(1)*4))

intd=80000000+torch.randn(20, dtype=torch.float64)*10000000
print([int(i) for i in intd])
print(intd.to(torch.long))
print(year_m[intd.to(torch.long)])
print(year_m[-1])
'''

edge_index = dataset.edge_index('paper', 'cites', 'paper')
print(f"type(edge_index) : {type(edge_index)}")
print(f"edge_index.shape : {edge_index.shape}")

row, col=edge_index
# row's year is bigger than col's year
# row cites col
small_row=[]
large_col=[]
for i in range(20000,30000):
    if year_m[row[i]]<year_m[col[i]]:
        small_row.append(year_m[row[i]])
        large_col.append(year_m[col[i]])
for i in range(len(small_row)):
    print(f"({small_row[i]}, {large_col[i]})", end=' ')
print()

