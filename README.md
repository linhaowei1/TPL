# Class Incremental Learning via Likelihood Ratio Based Task Prediction

This repository contains the code for our ICLR2024 paper [Class Incremental Learning via Likelihood Ratio Based Task Prediction](https://arxiv.org/abs/2309.15048) by [Haowei Lin](https://linhaowei1.github.io/), [Yijia Shao](https://shaoyijia.github.io/), Weinan Qian, Ningxin Pan, Yiduo Guo, and [Bing Liu](https://www.cs.uic.edu/~liub/).

**Update [2024.2.10]: Now we support DER++ and more pre-trained visual encoders!**

## Quick Links

- [Overview](#overview)
- [Requirements](#requirements)
- [Training](#training)
- [Evaluation](#evaluation)
- [Extension](#extension)
- [Bugs or Questions?](#bugs-or-questions)
- [Acknowledgements](acknowledgements#)
- [Citation](#citation)

## Overview

![](figures/TPL.png)

## Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org/). We run the experiments on Pytorch 2.0.1, and PyTorch version higher than `1.6.0` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

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

## Training

In the following section, we describe how to train the TPL model by using our code.

**Data**

Before training and evaluation, please download the datasets (CIFAR-10, CIFAR-100, TinyImageNet). The default working directory is set as ``~/data`` in our code. You can modify it according to your need.

**Pre-train Model**

We use the pre-train DeiT model provided by [MORE](https://github.com/k-gyuhak/MORE). Please download it and save the file as ``./ckpt/pretrained/deit_small_patch16_224_in661.pth``. If you would like to test other pre-trained visual encoders, also download to the same place (you can find the pre-trained weights in timm or huggingface). We provide the scripts for Dino, MAE, CILP, ViT (small, tiny), DeiT (small, tiny).

**Training scripts**

We provide the examplar training and evaluation script as `deit_small_in661.sh`. Just run the following command and you will get the results:

```bash
bash scripts/deit_small_in661.sh
```

This script performs both training and testing. The default training will train TPL for 5 random seeds. In training, the results will be logged in `ckpt` and the training results are $HAT_{CIL}$ without using TPLR inference techniques. After running evaluation, it will be replaced with new results. If you find you get a bad results, try to check if you run the `eval.py` accurately. The results for the first run with `seed=2023` will be saved in `./ckpt/seq0/seed2023/progressive_main_2023`.

For the results in the paper, we use Nvidia A100 GPUs with CUDA 11.7. Using different types of devices or different versions of CUDA/other software may lead to slightly different performance.

## Extension

Our repo also supports running baselines like DER++. If you are interested in other baselines, just follow the same way of DER++ to integrate your new code. Also, if you want to test TIL+OOD methods, you can just modify the inference code and include the OOD score computation in `baseline.py`. Our code base is vey extensible.

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email [Haowei](mailto:linhaowei@pku.edu.cn). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Acknowledgements

We thank [PyContinual](https://github.com/ZixuanKe/PyContinual) for providing an extendable framework for continual learning. We use their code structure as a reference when developing this code base.

## Citation

Please cite our paper if you use this code or part of it in your work:

```bibtex
@inproceedings{lin2023class,
      title={Class Incremental Learning via Likelihood Ratio Based Task Prediction}, 
      author={Haowei Lin and Yijia Shao and Weinan Qian and Ningxin Pan and Yiduo Guo and Bing Liu},
      year={2024},
      booktitle={International Conference on Learning Representations}
}
```
