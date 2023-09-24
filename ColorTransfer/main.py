import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.transform import resize
import os
from sklearn import cluster
import torch
import time
import argparse
from utils import *
import ot
np.random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='CT')
parser.add_argument('--num_iter', type=int, default=10000, metavar='N',
                    help='Num Interations')
parser.add_argument('--source', type=str, metavar='N',
                    help='Source')
parser.add_argument('--target', type=str, metavar='N',
                    help='Target')
parser.add_argument('--cluster',  action='store_true',
                    help='Use clustering')
parser.add_argument('--load',  action='store_true',
                    help='Load precomputed')
parser.add_argument('--palette',  action='store_true',
                    help='Show color palette')


args = parser.parse_args()


n_clusters = 3000
name1=args.source#path to images 1
name2=args.target#path to images 2
source = img_as_ubyte(io.imread(name1))
target = img_as_ubyte(io.imread(name2))
reshaped_target = img_as_ubyte(resize(target, source.shape[:2]))
name1=name1.replace('/', '')
name2=name2.replace('/', '')
if(args.cluster):
    X = source.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    source_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    source_k_means.fit(X)
    source_values = source_k_means.cluster_centers_.squeeze()
    source_labels = source_k_means.labels_

    # create an array from labels and values
    source_compressed = source_values[source_labels]
    source_compressed.shape = source.shape

    vmin = source.min()
    vmax = source.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Source")
    plt.imshow(source,  vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Source")
    plt.imshow(source_compressed.astype('uint8'),  vmin=vmin, vmax=vmax)
    os.makedirs('npzfiles', exist_ok=True)
    with open('npzfiles/'+name1+'source_compressed.npy', 'wb') as f:
        np.save(f, source_compressed)
    with open('npzfiles/'+name1+'source_values.npy', 'wb') as f:
        np.save(f, source_values)
    with open('npzfiles/'+name1+'source_labels.npy', 'wb') as f:
        np.save(f, source_labels)
    np.random.seed(0)

    X = target.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    target_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    target_k_means.fit(X)
    target_values = target_k_means.cluster_centers_.squeeze()
    target_labels = target_k_means.labels_

    # create an array from labels and values
    target_compressed = target_values[target_labels]
    target_compressed.shape = target.shape

    vmin = target.min()
    vmax = target.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Target")
    plt.imshow(target,  vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Target")
    plt.imshow(target_compressed.astype('uint8'),  vmin=vmin, vmax=vmax)

    with open('npzfiles/'+name2+'target_compressed.npy', 'wb') as f:
        np.save(f, target_compressed)
    with open('npzfiles/'+name2+'target_values.npy', 'wb') as f:
        np.save(f, target_values)
    with open('npzfiles/'+name2+'target_labels.npy', 'wb') as f:
        np.save(f, target_labels)
else:
    with open('npzfiles/'+name1+'source_compressed.npy', 'rb') as f:
        source_compressed = np.load(f)
    with open('npzfiles/'+name2+'target_compressed.npy', 'rb') as f:
        target_compressed = np.load(f)
    with open('npzfiles/'+name1+'source_values.npy', 'rb') as f:
        source_values = np.load(f)
    with open('npzfiles/'+name2+'target_values.npy', 'rb') as f:
        target_values = np.load(f)
    with open('npzfiles/'+name1+'source_labels.npy', 'rb') as f:
        source_labels = np.load(f)


for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
MaxSWcluster,MaxSW = transform_SW(source_values,target_values,source_labels,source,T=100,s_lr=0.1,sw_type='maxsw',num_iter=args.num_iter)
MaxSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
SWcluster,SW = transform_SW(source_values,target_values,source_labels,source,L=100,sw_type='sw',num_iter=args.num_iter)
SWtime = np.round(time.time() - start,2)




for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
vDSWcluster,vDSW = transform_SW(source_values,target_values,source_labels,source,L=50,T=2,s_lr=0.1,sw_type='vdsw',kappa=10,num_iter=args.num_iter)
vDSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)
start = time.time()
ISEBSWcluster,ISEBSW = transform_SW(source_values,target_values,source_labels,source,L=1,T=100,s_lr=0.1,f_type="exp",sw_type='isebsw',copy=False,num_iter=args.num_iter)
ISEBSWtime = np.round(time.time() - start,2)


# for _ in range(1000):
#     a = np.random.randn(100)
#     a = torch.randn(100)
# start = time.time()
# IMHEBSWcluster,IMHEBSW = transform_SW(source_values,target_values,source_labels,source,L=1,T=100,s_lr=0.1,f_type="exp",sw_type='imhebsw',copy=True,num_iter=args.num_iter)
# IMHEBSWtime = np.round(time.time() - start,2)
# for _ in range(1000):
#     a = np.random.randn(100)
#     a = torch.randn(100)
# start = time.time()
# SIREBSWcluster,SIREBSW = transform_SW(source_values,target_values,source_labels,source,L=1,T=100,s_lr=0.1,f_type="exp",sw_type='sirebsw',copy=True,num_iter=args.num_iter)
# SIREBSWtime = np.round(time.time() - start,2)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
source3=source_values.reshape(-1,3)
reshaped_target3=target_values.reshape(-1,3)

SWcluster=SWcluster/np.max(SWcluster)*255
MaxSWcluster=MaxSWcluster/np.max(MaxSWcluster)*255
vDSWcluster=vDSWcluster/np.max(vDSWcluster)*255
ISEBSWcluster=ISEBSWcluster/np.max(ISEBSWcluster)*255
# iMSWcluster=iMSWcluster/np.max(iMSWcluster)*255
# IMHEBSWcluster=IMHEBSWcluster/np.max(IMHEBSWcluster)*255
# SIREBSWcluster=SIREBSWcluster/np.max(SIREBSWcluster)*255
# viMSWcluster=viMSWcluster/np.max(viMSWcluster)*255



# f.suptitle("L={}, k={}, T={}".format(L, k, iter), fontsize=20)
C_SW = ot.dist(SWcluster,reshaped_target3)
C_MaxSW = ot.dist(MaxSWcluster, reshaped_target3)
C_vDSW = ot.dist(vDSWcluster, reshaped_target3)
C_ISEBSW = ot.dist(ISEBSWcluster, reshaped_target3)
# C_iMSW = ot.dist(iMSWcluster, reshaped_target3)
# C_IMHEBSW = ot.dist(IMHEBSWcluster, reshaped_target3)
# C_SIREBSW = ot.dist(SIREBSWcluster, reshaped_target3)
# C_viMSW = ot.dist(viMSWcluster, reshaped_target3)

W_SW = np.round(ot.emd2([],[],C_SW),2)
W_MaxSW = np.round(ot.emd2([], [], C_MaxSW),2)
W_vDSW = np.round(ot.emd2([], [], C_vDSW), 2)
W_ISEBSW = np.round(ot.emd2([], [], C_ISEBSW), 2)
# W_iMSW = np.round(ot.emd2([],[],C_iMSW),2)
# W_IMHEBSW = np.round(ot.emd2([],[],C_IMHEBSW),2)
# W_SIREBSW = np.round(ot.emd2([],[],C_SIREBSW),2)
# W_viMSW = np.round(ot.emd2([],[],C_viMSW),2)


f, ax = plt.subplots(1, 6, figsize=(12, 5))
ax[0].set_title('Source', fontsize=14)
ax[0].imshow(source)
# ax[1,0].scatter(source3[:, 0], source3[:, 1], source3[:, 2], c=source3 / 255)

ax[1].set_title('SW {}(s), $W_2={}$'.format(SWtime,W_SW), fontsize=12)
ax[1].imshow(SW)
# ax[1].scatter(SWcluster[:, 0], SWcluster[:, 1], SWcluster[:, 2], c=SWcluster / 255)

ax[2].set_title('Max-SW {}(s), $W_2={}$'.format(MaxSWtime,W_MaxSW), fontsize=12)
ax[2].imshow(MaxSW)
# ax[1,2].scatter(MaxSWcluster[:, 0], MaxSWcluster[:, 1], MaxSWcluster[:, 2], c=MaxSWcluster / 255)

ax[3].set_title('v-DSW {}(s), $W_2={}$'.format(vDSWtime,W_vDSW), fontsize=12)
ax[3].imshow(vDSW)
# ax[1,3].scatter(vDSWcluster[:, 0], vDSWcluster[:, 1], vDSWcluster[:, 2], c=vDSWcluster / 255)



ax[4].set_title('IS-EBSWprivate-e {}(s), $W_2={}$'.format(ISEBSWtime,W_ISEBSW), fontsize=12)
ax[4].imshow(ISEBSW)
# ax[3,0].scatter(ISEBSWcluster[:, 0], ISEBSWcluster[:, 1], ISEBSWcluster[:, 2], c=ISEBSWcluster / 255)

# ax[2,1].set_title('IMH-EBSWprivate-e {}(s), $W_2={}$'.format(IMHEBSWtime,W_IMHEBSW), fontsize=12)
# ax[2,1].imshow(IMHEBSW)
# ax[3,1].scatter(IMHEBSWcluster[:, 0], IMHEBSWcluster[:, 1], IMHEBSWcluster[:, 2], c=IMHEBSWcluster / 255)
#
# ax[2,2].set_title('SIR-EBSWprivate-e {}(s), $W_2={}$'.format(SIREBSWtime,W_SIREBSW), fontsize=12)
# ax[2,2].imshow(SIREBSW)
# ax[3,2].scatter(SIREBSWcluster[:, 0], SIREBSWcluster[:, 1], SIREBSWcluster[:, 2], c=SIREBSWcluster / 255)



ax[5].set_title('Target', fontsize=14)
ax[5].imshow(reshaped_target)
# ax[3,3].scatter(reshaped_target3[:, 0], reshaped_target3[:, 1], reshaped_target3[:, 2], c=reshaped_target3 / 255)

for i in range(6):
    # for j in range(4):
        ax[i].get_yaxis().set_visible(False)
        ax[i].get_xaxis().set_visible(False)

plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=0.88, bottom=0.01, wspace=0, hspace=0.145)
plt.show()
plt.clf()
plt.close()
#####
# f, ax = plt.subplots(2, 4, figsize=(12, 5))
# # f.suptitle("m={}, k={}, T={}".format(m, k, iter), fontsize=20)
# ax[0,0].set_title('Source', fontsize=14)
# ax[0,0].imshow(source)
#
#
# ax[0,1].set_title('SW {}(s), $W_2={}$'.format(SWtime,W_SW), fontsize=12)
# ax[0,1].imshow(SW)
#
#
# ax[0,2].set_title('Max-SW {}(s), $W_2={}$'.format(MaxSWtime,W_MaxSW), fontsize=12)
# ax[0,2].imshow(MaxSW)
#
#
# ax[0,3].set_title('v-DSW  {}(s), $W_2={}$'.format(vDSWtime,W_vDSW), fontsize=12)
# ax[0,3].imshow(vDSW)
#
#
#
#
# ax[1,0].set_title('IS-EBSWprivate-e {}(s), $W_2={}$'.format(ISEBSWtime,W_ISEBSW), fontsize=12)
# ax[1,0].imshow(ISEBSW)
#
# ax[1,1].set_title('IMH-EBSWprivate-e {}(s), $W_2={}$'.format(IMHEBSWtime,W_IMHEBSW), fontsize=12)
# ax[1,1].imshow(IMHEBSW)
#
# ax[1,2].set_title('SIR-EBSWprivate-e {}(s), $W_2={}$'.format(SIREBSWtime,W_SIREBSW), fontsize=12)
# ax[1,2].imshow(SIREBSW)
#
#
#
# ax[1,3].set_title('Target', fontsize=14)
# ax[1,3].imshow(reshaped_target)
#
#
# for i in range(2):
#     for j in range(4):
#         ax[i,j].get_yaxis().set_visible(False)
#         ax[i,j].get_xaxis().set_visible(False)
#
# plt.tight_layout()
# plt.subplots_adjust(left=0, right=1, top=0.88, bottom=0.01, wspace=0, hspace=0.145)
# plt.show()



