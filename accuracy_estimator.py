
import os.path as osp
import sys
import time
sys.path.insert(0,'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park')    
import numpy as np
import torch
import torch.nn.functional as F
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from torch_sparse import SparseTensor
from tqdm import tqdm

ROOT='/fs/ess/PAS1289/mag240m_kddcup2021'
dataset = MAG240MDataset(root = ROOT)
valid_idx=dataset.get_idx_split('valid')
paper_label=dataset.paper_label[valid_idx]

def calculate_val_acc(y_preds):
    cnt=len(paper_label)
    acc_cnt=0
    for i in range(len(paper_label)):
        if(y_preds[i]==paper_label[i]):
            acc_cnt+=1
    return acc_cnt/cnt


t0=time.time()
dataset = MAG240MDataset(root = ROOT)
paper_edge_index = dataset.edge_index('paper','paper')
author_edge_index = dataset.edge_index('author', 'writes', 'paper')
rowp, colp = torch.from_numpy(paper_edge_index)
rowa, cola = torch.from_numpy(author_edge_index)
paper_adj_t = SparseTensor(row=rowp, col=colp, is_sorted=True)
author_adj_t = SparseTensor(row=rowa, col=cola, is_sorted=True)
paper_rowcount=paper_adj_t.storage.rowcount()
author_rowcount=author_adj_t.storage.colcount()
print(f"Loading done! : {time.time()-t0}s")


pred_dirs=["/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/mono-NS_p=0.1_batch=1024.npy",
        "/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/New-time-NS_p=0.1_batch=1024.npy",
        "/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/val_activation/toggle-NS_p=0.1_batch=1024.npy"]

for prediction_dir in pred_dirs:
    name=prediction_dir.split('/')[-1][:-4]
    f_log=open(f'/users/PAS1289/oiocha/OGB-NeurIPS-Team-Park/Analysis/prediction_stat_[{name}].txt','w+')


    # Shape : (138949,153)
    cur_activation=np.load(prediction_dir).astype(np.float16)
    y_preds = np.argmax(cur_activation, axis=1)
    predict_acc=calculate_val_acc(y_preds)


    # Count label distribution
    ground_cnt=[0]*153
    predict_cnt=[0]*153
    paper_degree_sum=[0]*153
    author_degree_sum=[0]*153
    correct=[0]*153
    wrong=[0]*153

    for i, label in enumerate(paper_label):
        ground_cnt[int(label)]+=1
        paper_degree_sum[int(label)]+=int(paper_rowcount[valid_idx[i]])
        author_degree_sum[int(label)]+=int(author_rowcount[valid_idx[i]])
        if(paper_label[i]==y_preds[i]):
            correct[int(paper_label[i])]+=1
        else:
            wrong[int(paper_label[i])]+=1
    for label in y_preds:
        predict_cnt[int(label)]+=1


    for i in range(153):
        f_log.write(f"{i}th label ({(correct[i]+wrong[i])/len(paper_label)*100:.2f}%) | {(correct[i]/(correct[i]+wrong[i])*100) if correct[i]+wrong[i]!=0 else 0:.2f}% AC/(AC+WA), {correct[i]/(predict_cnt[i])*100 if predict_cnt[i]!=0 else 0:.2f}% AC/CL | Mean degree. Paper:{paper_degree_sum[i]/ground_cnt[i] if ground_cnt[i]!=0 else 0:.2f}, Author:{author_degree_sum[i]/ground_cnt[i] if ground_cnt[i]!=0 else 0:.2f} | {correct[i]}AC, {wrong[i]}WA, {predict_cnt[i]}CL.\n")

    mx_correct=0
    for i in range(len(ground_cnt)):
        mx_correct+=min(ground_cnt[i], predict_cnt[i])

    print(f"Prediction accuracy : {predict_acc:.5f}")
    f_log.write(f"Prediction accuracy : {predict_acc:.5f}\n")
    print(f"Maximum accuracy : {mx_correct/len(paper_label):.5f}")
    f_log.write(f"Maximum accuracy : {mx_correct/len(paper_label):.5f}\n")

    f_log.flush()