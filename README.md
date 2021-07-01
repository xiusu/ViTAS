# Vision Transformer Architecture Search

This repository open source the code for ViTAS: Vision Transformer Architecture Search.

## Requirements
1. torch>=1.4.0
1. torchvision
1. pymoo==0.3.0 for evaluation --> pip install pymoo==0.3.0 --user
1. change the 'data_dir' in yaml from search/retrain/inference directory to your ImageNet data path, note that each yaml have four 'data_dir' for training the supernet (train data), evolutionary sampling with supernet (val data), retraining the searched architecture (train data), and test the trained architecture (test data).
1. This code is based on slurm for distributed training.

## Reproducing

### To implement the search with ViTAS.

The supernet training process of ViTAS will be updated within two weeks after a detailed test.

We will update more information about ViTAS, please stay tuned on this repository.

### To retrain our searched models.
For example, train our 1.3G architecture searched by ViTAS.
```
chmod +x ./script/command.sh

chmod +x ./script/vit_1.3G_retrain.sh

./script/vit_1.3G_retrain.sh
```

### To inference our searched results.

For example, inference our 1.3G architecture searched by ViTAS.
```
chmod +x ./script/command.sh

chmod +x ./script/vit_1.3G_inference.sh

./script/vit_1.3G_inference.sh
```

## Results of searched architectures with ViTAS

Model name | FLOPs | Top 1 | Top 5 | Download
------------ | ------------- | ------------- | ------------- | -------------
ViTAS-A | 858M | 71.1% | 89.8% | xxx
ViTAS-B | 1.0G | 72.4% | 90.6% | xxx
ViTAS-C | 1.3G | 74.7% | 92.0% | xxx 
ViTAS-E | 2.7G | 77.4% | 93.8% | xxx
ViTAS-F | 4.9G | 80.6% | 95.1% | xxx

## Citation

If you find that ViTAS interesting and help your research, please consider citing it:

```
@misc{su2021vision,
      title={Vision Transformer Architecture Search}, 
      author={Xiu Su and Shan You and Jiyang Xie and Mingkai Zheng and Fei Wang and Chen Qian and Changshui Zhang and Xiaogang Wang and Chang Xu},
      year={2021},
      eprint={2106.13700},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
