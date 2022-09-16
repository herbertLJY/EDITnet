# EDITnet

This is the PyTorch implementation of the models in **EDITnet: A Lightweight Network for Unsupervised Domain Adaptation in Speaker Verification**, 
INTERSPEECH2022.

By Jingyu Li, Wei Liu, Tan Lee.

[Paper-pdf](https://arxiv.org/pdf/2206.07548v1.pdf)

## The CVAE structure
This code only includes the structure of the CVAE network, for training and evaluation. You could utilize it with your 
own embedding extraction model.

![Image](VAE.pdf)

## Before training
Please calculate the mean and std of the embeddings by your trained model, on the target and source domain datasets, 
respectively. 
They are used for normalizing the CVAE inputs:
```
self.feat_vox_train_mean = feat_vox_train_mean  # please calculate it on your trained model and dataset
self.feat_vox_train_std = feat_vox_train_std    # please calculate it on your trained model and dataset
self.feat_cn_train_mean = feat_cn_train_mean    # please calculate it on your trained model and dataset
self.feat_cn_train_std = feat_cn_train_std      # please calculate it on your trained model and dataset
```

## Citation
If you find this work helpful in your project, please consider citing:
```
@article{li2022editnet,
  title={EDITnet: A Lightweight Network for Unsupervised Domain Adaptation in Speaker Verification},
  author={Li, Jingyu and Liu, Wei and Lee, Tan},
  journal={arXiv preprint arXiv:2206.07548},
  year={2022}
}

```