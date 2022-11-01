import numpy as np
import matplotlib.pyplot as plt
import time
import os
import os.path as osp
import sys
import torch
t0=time.time()

dirs=[
    "/users/PAS1289/oiocha/results/b1/acua_ensemble/y_pred_mag240m_test-challenge.npz",
    "/users/PAS1289/oiocha/results/b2/acua_ensemble/y_pred_mag240m_test-challenge.npz",
    "/users/PAS1289/oiocha/results/b3/acua_ensemble/y_pred_mag240m_test-challenge.npz",
    "/users/PAS1289/oiocha/results/b4/acua_ensemble/y_pred_mag240m_test-challenge.npz",
    "/users/PAS1289/oiocha/results/b5/acua_ensemble/y_pred_mag240m_test-challenge.npz",
]
'''
for i in range(5):
    dir=dirs[i]
    y_preds=np.load(dir)['y_pred']
    #print(type(y_preds))
    #print(y_preds.shape)

    cnt=np.zeros(58726).astype(np.int)
    for ii in y_preds:
        cnt[ii]+=1

    valid_label_prob=torch.load('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/mingi_valid_label.pt')
    #print(valid_label_prob.shape)

    l2_error=0
    l1_error=0
    for ii in range(153):
        #print(f"{i}th label cnt : {cnt[i]}, prob : {cnt[i]/58726:.5f}, valid prob : {valid_label_prob[i]:.5f}")
        l2_error+=valid_label_prob[ii]*(valid_label_prob[ii]-cnt[ii]/58726)**2
        l1_error+=valid_label_prob[ii]*abs(valid_label_prob[ii]-cnt[ii]/58726)

    print(f"{i}th block", end=' ')
    print(f"l2_error : {l2_error}, l1_error : {l1_error}")
'''
dir="/users/PAS1289/oiocha/results/acua_ensemble/y_pred_mag240m_test-challenge.npz"
y_preds=np.load(dir)['y_pred']
#print(type(y_preds))
#print(y_preds.shape)

cnt=np.zeros(58726).astype(np.int)
for ii in y_preds:
    cnt[ii]+=1

valid_label_prob=torch.load('/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/mingi_valid_label.pt')
#print(valid_label_prob.shape)

l2_error=0
l1_error=0
for ii in range(153):
    #print(f"{i}th label cnt : {cnt[i]}, prob : {cnt[i]/58726:.5f}, valid prob : {valid_label_prob[i]:.5f}")
    l2_error+=valid_label_prob[ii]*(valid_label_prob[ii]-cnt[ii]/58726)**2
    l1_error+=valid_label_prob[ii]*abs(valid_label_prob[ii]-cnt[ii]/58726)

print(f"0th block", end=' ')
print(f"l2_error : {l2_error}, l1_error : {l1_error}")
