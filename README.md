# OGB-NeurIPS challenge 2022

## Description
It is multi-label classification of ArXiv papers on MAG240M dataset, which is Transductive Knowledge Graph.
Since the graph is huge and data is splited by time, three main challenges have appeared.


### Challenges
- Label distributions that vary significantly over time.
- 170 times as many unlabeled node as labeled node (Computational intensity)
- Train node / Valid node / Test node all have different nature. (e.g. Test node has little ‘cited’ link)


### Methods
I have mainly used RGNN, with several modification to deal mentioned problems.
- relation.py : Sampling by relation.
- baidu.py : Concatenate masked label with input feature. Combine GNN message passing and Label propagation is proposed in Shi et al.(2020)
- toggle.py : Toggle sampling, which sample only ‘cite’ edge in first sampling, after that, sample both direction.
- meta.py : Meta sampling, which sample only
- bi_toggle.py : Separate paper-cite-paper and paper-cited-paper link, and using previous embedding as new feature.
- acua.py : cross-validation version of bi_toggle.py
Additionally, linear.py is linear model that do not use graph information, is used as basaline experiment.

In our experiment, acua.py with 5-cross validation performs best. (Validation accuracy : 0.7302)



## Usage
### Single model (without ensemble)
Train.
```
python OGB-NeurIPS-Team-Park/bi_full.py --label_disturb_p=0.1 --batch_size=1024 
```

Inference
```
python OGB-NeurIPS-Team-Park/bi_full.py --label_disturb_p=0.0 --time_disturb_p=0.0 --batch_size=1024 --evaluate --ckpt=/users/PAS1289/oiocha/logs/bi-full_p=0.1_batch=1024/lightning_logs/version_13082877/checkpoints/epoch=34-step=38044.ckpt
```


## Citing OGB / OGB-LSC
If you use OGB or [OGB-LSC](https://ogb.stanford.edu/docs/lsc/) datasets in your work, please cite our papers (Bibtex below).
```
@article{hu2020ogb,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  journal={arXiv preprint arXiv:2005.00687},
  year={2020}
}
```
```
@article{hu2021ogblsc,
  title={OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Ren, Hongyu and Nakata, Maho and Dong, Yuxiao and Leskovec, Jure},
  journal={arXiv preprint arXiv:2103.09430},
  year={2021}
}
```
