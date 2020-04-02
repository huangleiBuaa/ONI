# Pytorch Experiments
This directory/project (Pytorch implementation) provides the ImageNet classification and GAN experiments of the following paper:

**Controllable Orthogonalization in Training DNNs. Lei Huang, Li Liu, Fan Zhu, Diwen Wan, Zehuan Yuan, Bo Li, Ling Shao. CVPR 2020 (accepted)**
# ONI

## Requirements and Dependency
* Install [PyTorch](http://torch.ch) with CUDA (for GPU). (Experiments are validated on python 3.6.8 and pytorch 1.0.1)
* (For visualization if needed), install the dependency [visdom](https://github.com/facebookresearch/visdom) by:
```Bash
pip install visdom
 ```


## Experiments
 
 #### 1. ImageNet experiments.

The scripts  are in `./ImageNet/experiments/`. For example, one can  run the ONI on 18 layer residual network without BN,  by following script: 
  ```Bash
bash resnet18_NoBN_ONI.sh
 ```
**Note: 1) change the hyper-parameter `--dataset-root` to your ImageNet dataset path; 
2) change the `CUDA_VISIBLE_DEVICES`, if your machine has less GPUs than the value. (e.g., you have to set the value as 0 if you have only one GPU on your machine).**

 #### 2. GAN.
The scripts  are in `./GAN/experiments/`. For example, one can  run the ONI on ResNet with bce loss,  by following script: 
  ```Bash
   bash execute_ONI_resnet_bce.sh
 ```
 
## Acknowledgement

* The ImageNet classification is based on the [IterNorm-pytorch](https://github.com/huangleiBuaa/IterNorm-pytorch)
* The GAN experiments are based on the [Pytorch implementation of spectral normalization](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan), where the calculation of FID and IS is based on the [Pytorch implementation of BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
