# EBSW
Official PyTorch implementation for paper:  [Energy-Based Sliced Wasserstein Distance](https://arxiv.org/abs/2304.13586)

Details of the model architecture and experimental results can be found in our papers.

```
@article{nguyen2023energy,
  title={Energy-Based Sliced Wasserstein Distance},
  author={Khai Nguyen and Nhat Ho},
  journal={Advances in Neural Information Processing Systems},
  year={2023},
  pdf={https://arxiv.org/pdf/2304.13586.pdf}
}
```
Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by [Khai Nguyen](https://khainb.github.io).


## What is included?
* Point-Cloud Gradient flow 
* Color Transfer
* Deep Point-Cloud Reconstruction


## Point-Cloud Gradient flow 
```
cd GradientFlow
python main_point.py
```

## Color Transfer

```
cd ColorTransfer
python main.py --source [source image] --target [target image] --num_iter 2000 --cluster

```

## Deep Point-cloud Reconstruction
Please read the README file in the PointcloudAE folder.

## Acknowledgment
The structure of this repo is largely based on [PointSWD](https://github.com/VinAIResearch/PointSWD). The structure of folder `render` is largely based on [Mitsuba2PointCloudRenderer](https://github.com/tolgabirdal/Mitsuba2PointCloudRenderer). The implementation of the Von Mises-Fisher distribution is taken from [s-vae-pytorch](https://github.com/nicola-decao/s-vae-pytorch).