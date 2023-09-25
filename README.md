# Class Incremental Learning via Likelihood Ratio Based Task Prediction

This repository contains the code for our paper *Class Incremental Learning via Likelihood Ratio Based Task Prediction*.

## Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org/). Please use the correct 1.6.0 version corresponding to your platforms/CUDA versions to faithfully reproduce our results. PyTorch version higher than `1.6.0` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```
pip install torch==1.6.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```
pip install torch==1.6.0
```

Then run the following script to install the remaining dependencies,

```
pip install -r requirements.txt
```

**Attention**: Our model is based on `timm==0.4.12`. Using them from other versions may cause some unexpected bugs.

## Train TPLR

In the following section, we describe how to train the TPLR model by using our code.

**Data**

Before training and evaluation, please download the datasets (CIFAR-10, CIFAR-100, TinyImageNet). The default working directory is set as ``~/data`` in our code. You can modify it according to your need.

**Pre-train Model**

We use the pre-train DeiT model provided by [MORE](https://github.com/k-gyuhak/MORE). Please download it and save the file as ``./deit_pretrained/best_checkpoint.pth``

**Training scripts**

We provide all the example training scripts to run TPLR. e.g., for C10-5T, train the network using this command:

```bash
bash scripts/deit_C10_5T.sh
```

For the results in the paper, we use Nvidia GeForce RTX2080Ti GPUs with CUDA 10.2. Using different types of devices or different versions of CUDA/other software may lead to slightly different performance.

### Evaluation

Continue with the C10-5T example. Once you finished training, come back to the root directory and simply run this command:

```bash
bash scripts/deit_C10_5T_eval.sh
```

The results for the first sequence with `seed=2023` will be saved in `./data/seq0/seed2023/progressive_main_2023`.