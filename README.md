# OGB-NeurIPS challenge 2022

## Description
It is multi-label classification of ArXiv papers on transductive knowledge graph(MAG240M).

<img  align="center" width="500" alt="image" src="https://user-images.githubusercontent.com/100084401/199390976-887f7b28-0e51-4044-ad3f-6d39c35a618f.png">
 
Since the graph is huge and data is splited by time, three main challenges have appeared.


### Challenges
- Label distributions that vary significantly over time.
- 170 times as many unlabeled node as labeled node (Computationally very inefficient)
- Train node / Valid node / Test node all have different nature. (e.g. Test node has little ‘cited’ link)



### Methods
I have mainly used RGNN, with several modification to deal mentioned problems.
- relation.py : Sampling by relation, added time positional encoding.
- baidu.py : Concatenate masked label with input feature. Combine GNN message passing and Label propagation is proposed in Shi et al.(2020)
- toggle.py : Toggle sampling, which sample only ‘cite’ edge in first sampling, after that, sample both direction.
- meta.py : Meta sampling, which sample only
- bi_toggle.py : Separate paper-cite-paper and paper-cited-paper link, and using previous embedding as new feature.
- acua.py : Cross-validation version of bi_toggle.py
Additionally, linear.py is linear model that do not exploit graph information, is used as basaline experiment.

As a result of extensive experimentation, the use of embeddings obtained using similar models adversely affected the effectiveness. Also, toggle sampling was more effective when the model complexity was small, but full sampling achieved slightly higher accuracy when constructing a very large model. We reflected the results of these experiments when training the final model(acua.py with 5-cross validation, accuracy = 0.7302)



## Usage
### Single model (without ensemble)
Train
```
python OGB-NeurIPS-Team-Park/bi_full.py --label_disturb_p=0.1 --batch_size=1024 
```

Inference
```
python OGB-NeurIPS-Team-Park/bi_full.py --label_disturb_p=0.0 --time_disturb_p=0.0 --batch_size=1024 --evaluate --ckpt=/users/PAS1289/oiocha/logs/bi-full_p=0.1_batch=1024/lightning_logs/version_13082877/checkpoints/epoch=34-step=38044.ckpt
```
I trained 35 epochs, and the model with the highest accuracy among them was used for inference.



### Multi model
Train
```
python OGB-NeurIPS-Team-Park/acua.py --link=full --cross_partition_number=5 --cross_partition_idx=0
python OGB-NeurIPS-Team-Park/acua.py --link=full --cross_partition_number=5 --cross_partition_idx=1
python OGB-NeurIPS-Team-Park/acua.py --link=full --cross_partition_number=5 --cross_partition_idx=2
python OGB-NeurIPS-Team-Park/acua.py --link=full --cross_partition_number=5 --cross_partition_idx=3
python OGB-NeurIPS-Team-Park/acua.py --link=full --cross_partition_number=5 --cross_partition_idx=4
```

Inference
```
python OGB-NeurIPS-Team-Park/acuatest.py --cross_partition_idx=0 --ckpt=/users/PAS1289/oiocha/logs/acua_full_p=0.1_block=0/lightning_logs/version_13341040/checkpoints/epoch=32-step=39328.ckpt
python OGB-NeurIPS-Team-Park/ensemble_test.py
```
acuatest.py is a test-challenge inference code.

ensemble_test.py makes soft-vote for 5 ensemble models(15 inferences total) and save submission file. In order for the code to work, you need to modify the dir_list_test inside the file appropriately.

Because the size of the validation set is small, the accuracy fluctuates greatly, so it is not recommended to use the highest accuracy model. Instead, I used the following empirical rule: The activations obtained from the 28, 30, and 32 epochs models are averaged and used for inference.

For each model, training takes 70 hours on 2x Intel Xeon Platinum 8268(48 cores, 768GB memory) with no GPUs.
Inference time for each model takes no longer than 10 minutes.




## References
[1] Addanki, Ravichandra et al. “LARGE-SCALE NODE CLASSIFICATION WITH BOOTSTRAPPING.” (2021).

[2] Hu, Weihua, et al. "Ogb-lsc: A large-scale challenge for machine learning on graphs." arXiv preprint arXiv:2103.09430 (2021).

[3] Hu, Weihua, et al. "Open graph benchmark: Datasets for machine learning on graphs." Advances in neural information processing systems 33 (2020): 22118-22133.

[4] Preprint, A et al. “R-UNIMP: SOLUTION FOR KDDCUP 2021 MAG240M-LSC.” (2021).

[5] Veličković, Petar, et al. "Graph attention networks." arXiv preprint arXiv:1710.10903 (2017).

[6] Velickovic, Petar, et al. "Deep Graph Infomax." ICLR (Poster) 2.3 (2019): 4.
