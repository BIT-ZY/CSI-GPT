## Overview 
This is the PyTorch implementation of the paper [CSI-GPT: Integrating Generative Pre-Trained Transformer with Federated-Tuning to Acquire Downlink Massive MIMO Channels](https://arxiv.org/abs/2406.03438).
If you feel this repo helpful, please cite our paper: 
``` 
@inproceedings{Zeng2024CSIGPTIG,
  title={CSI-GPT: Integrating Generative Pre-Trained Transformer with Federated-Tuning to Acquire Downlink Massive MIMO Channels},
  author={Ye Zeng and Li Qiao and Zhen Gao and Tong Qin and Zhonghuai Wu and Sheng Chen and Mohsen Guizani},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:270257824} }
```
## Requirements 
To use this project, you need to ensure the following requirements are installed.
- Python >= 3.8
- Pytorch >= 1.11.0
## Data Preparation 
The CDL model is generated from [Sionna](https://developer.nvidia.cn/sionna/).
## Folder 
The folder [SWTCAN] corresponds to the [IV.SIMULATION RESULTS part A] code in the article.

The folder [VAE-CSG] corresponds to the [IV.SIMULATION RESULTS part B] code in the article.

The folder [Fderated Tuning] corresponds to the [IV.SIMULATION RESULTS part C] code in the article.
