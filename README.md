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
In each yaml, the 'save_path' in 'search' controls all paths (eg., line 34 in inference/ViTAS_1.3G_inference.yaml). The code will automatically build the path of 'save_path'+'search/checkpoint/' for your supernet, and also 'save_path' + 'retrain/checkpoint' for retraining the searched architecture.

Therefore, to inference the provided pth file, you need to build a path of 'save_path/retrain/checkpoint/download.pth' ('save_path' specified in yaml and download.pth provided in below table).

The extract code for Baidu Cloud is 'c7gn'.



Model name | FLOPs | Top 1 | Top 5 | Download
------------ | ------------- | ------------- | ------------- | -------------
ViTAS-A | 858M | 71.1% | 89.8% | [Google Drive](https://drive.google.com/drive/folders/15xGXCBXlmvQgFyw4qFHw2-Rx6M-5JS0U?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1zl2c2AicGI60QaSpDwtusw)
ViTAS-B | 1.0G | 72.4% | 90.6% | [Google Drive](https://drive.google.com/drive/folders/1Hwt2rj4GWZsMLq8zCBMPX0TKe7-owWoU?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1zl2c2AicGI60QaSpDwtusw)
ViTAS-C | 1.3G | 74.7% | 92.0% | [Google Drive](https://drive.google.com/drive/folders/151xZk-v6bLtZuzqxmoSagtehb2e5JpSM?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1zl2c2AicGI60QaSpDwtusw)
ViTAS-E | 2.7G | 77.4% | 93.8% | [Google Drive](https://drive.google.com/drive/folders/1JwW5xTObaAosFsNZErkiND_rDnj6SEuG?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1zl2c2AicGI60QaSpDwtusw)
ViTAS-F | 4.9G | 80.6% | 95.1% | [Google Drive](https://drive.google.com/drive/folders/11gpbIr4b7NJU14lIYvU5deRYHHeOFS1B?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1zl2c2AicGI60QaSpDwtusw)

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
