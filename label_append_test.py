import numpy as np

train_idx=[0,10,20,30]
train_label=[0,3,6,9] # 10 classes
feature = np.random.random((50,15)).astype(np.float16) # 50*10
print(feature.shape)

encoding = np.eye(10)[train_label].astype(np.float16)
print(encoding)
'''
append_feat=np.zeros((50,10)).astype(np.float16)
append_feat[train_idx]=encoding
print(append_feat.shape)

print(append_feat[train_idx])
print(append_feat[0])
print(append_feat[1])

total_feat=np.concatenate((feature,append_feat), axis=1)
print(total_feat.shape)
print(total_feat)

for i in append_feat:
    for j in i:
        if type(j)!=np.float16:
            print(j)
'''
randidx=np.random.randint(0,50,10)
print(type(randidx[0]))
print(feature[randidx])

idx=[3,5,2]
val=[1,2,3]
dic={}
for i in range(3):
    dic[idx[i]]=val[i]

for i in range(10):
    if i in dic:
        print(i, dic[i])