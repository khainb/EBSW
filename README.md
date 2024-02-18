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

## Requirements
To install the required python packages, run
```
pip install -r requirements.txt
```

## What is included?
* Implementation of EBSW
* Point-Cloud Gradient flow 
* Color Transfer
* Deep Point-Cloud Reconstruction

## Implementation of EBSW

We recommend IS-EBSW-e as the default implementation of EBSW. For other variants, we refer to our implementation in experiments.

```
import torch
def rand_projections(dim, num_projections=1000,device='cpu'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance

def ISEBSW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    wasserstein_distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    wasserstein_distances =  wasserstein_distances.view(1,L)
    weights = torch.softmax(wasserstein_distances,dim=1)
    sw = torch.sum(weights*wasserstein_distances,dim=1).mean()
    return  torch.pow(sw,1./p)
```

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