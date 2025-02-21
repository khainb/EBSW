#### Table of Content

- [Energy-Based Sliced Wasserstein distance]()
  - [Getting Started](#getting-started)
    - [Datasets](#datasets)
      - [ShapeNet Core with 55 categories (refered from FoldingNet.)](#shapenet-core-with-55-categories-refered-from-foldingnet)
      - [ModelNet40](#modelnet40)
      - [ShapeNet Chair](#shapenet-chair)
      - [3DMatch](#3dmatch)
    - [Installation](#installation)
    - [For docker users](#for-docker-users)
  - [Experiments](#experiments)
    - [Point-cloud reconstruction](#point-cloud-reconstruction)
  - [Acknowledgment](#acknowledgment)

  

## Getting Started
### Datasets
#### ShapeNet Core with 55 categories (refered from <a href="http://www.merl.com/research/license#FoldingNet" target="_blank">FoldingNet</a>.)
```bash
  cd dataset
  bash download_shapenet_core55_catagories.sh
```
#### ModelNet40
```bash
  cd dataset
  bash download_modelnet40_same_with_pointnet.sh
```
#### ShapeNet Chair
```bash
  cd dataset
  bash download_shapenet_chair.sh
``` 
#### 3DMatch
```bash
  cd dataset
  bash download_3dmatch.sh
```
### Installation
The code has been tested with Python 3.6.9, PyTorch 1.2.0, CUDA 10.0 on Ubuntu 18.04.  

To install the required python packages, run
```bash
pip install -r requirements.txt
```

To compile CUDA kernel for CD/EMD loss:
```bash
cd metrics_from_point_flow/pytorch_structural_losses/
make clean
make
```

### For docker users
For building the docker image simply run the following command in the root directory
```bash
docker build -f Dockerfile -t <tag> .
```

## Experiments
### Point-cloud reconstruction
Available arguments for training an autoencoder
```bash
train.py [-h] [--config CONFIG] [--logdir LOGDIR]
                [--data_path DATA_PATH] [--loss LOSS]
                [--autoencoder AUTOENCODER]

optional arguments:
  -h, --help                  show this help message and exit
  --config CONFIG             path to json config file
  --logdir LOGDIR             path to the log directory
  --data_path DATA_PATH       path to data for training
  --loss LOSS                 loss function. One of [swd, msw, vsw, ebsw]
  --autoencoder AUTOENCODER   model name. One of [pointnet]
  --f_type                    energy function type [exp, identity]
  --estimation_type           estimation of EBSWprivate [IS,SIR,IMH,RMH]
  --gradient_type             gradient estimator [independent (copy), normal (reinforce)]
  --inter_dim                 dimension of keys
  --kappa                     scale of vMF distribution
```

Example
```bash
python3 train.py --config="config.json" \
                --logdir="logs/" \
                --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz" \
                --loss="swd" \
                --autoencoder="pointnet"

# or in short, you can run
bash train.sh
```

To test reconstruction
```bash
python3 reconstruction/reconstruction_test.py  --config="reconstruction/config.json" \
                                              --logdir="logs/" \
                                              --data_path="dataset/modelnet40_ply_hdf5_2048/"

# or in short, you can run
bash reconstruction/test.sh
```

To render input and reconstructed point-clouds, please follow the instruction in `render/README.md` to install dependencies. To reproduce Figure 4 in our paper, run the following commands *after training all autoencoders*
```bash
python3 save_point_clouds.py
cd render
bash render_reconstruction.sh
cd ..
python3 concat_reconstructed_images.py
```


## Acknowledgment
The structure of this repo is largely based on [PointSWD](https://github.com/VinAIResearch/PointSWD). The structure of folder `render` is largely based on [Mitsuba2PointCloudRenderer](https://github.com/tolgabirdal/Mitsuba2PointCloudRenderer). We are very grateful for their open sources.