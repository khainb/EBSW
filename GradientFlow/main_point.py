
import ot

##############################################
# Setup
# ---------------------
import random
import time
from utils import *
import numpy as np
import torch
for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
A = np.load("reconstruct_random_50_shapenetcore55.npy")
ind1=31
ind2=30
target=A[31]
source=A[30]
device='cpu'
f_type='exp'
learning_rate = 0.0001
N_step=500
eps=0
L=10
print_steps = [0,99,199,299,399,499]
Y = torch.from_numpy(target)
N=target.shape[0]
copy=False



for L in [10,100]:
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("SW {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)

            optimizer.zero_grad()
            sw= SW(X,Y,L=L)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().data.numpy())
        np.save("saved/SW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/SW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/SW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")




for s_lr in [0.1,0.01]:
    for L in [10,100]:
        for seed in [1,2,3]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            points=[]
            caltimes=[]
            distances=[]
            X=torch.tensor(source, requires_grad=True)
            optimizer = torch.optim.SGD([X], lr=learning_rate)

            start = time.time()
            for i in range(N_step):
                if (i in print_steps):
                    distance, cal_time = compute_true_Wasserstein(X, Y), time.time() - start
                    print("Max-SW {}:{} ({}s)".format(i+1,compute_true_Wasserstein(X,Y),np.round(time.time()-start,2)))
                    points.append(X.clone().data.numpy())
                    caltimes.append(cal_time)
                    distances.append(distance)
                optimizer.zero_grad()
                sw= N*MaxSW(X,Y,s_lr=s_lr,n_lr=L)
                sw.backward()
                optimizer.step()
            points.append(Y.clone().data.numpy())
            np.save("saved/Max-SW_T{}_lr{}_points_seed{}.npy".format(L,s_lr,seed),np.stack(points))
            np.savetxt("saved/Max-SW_T{}_lr{}_distances_seed{}.txt".format(L,s_lr,seed), np.array(distances), delimiter=",")
            np.savetxt("saved/Max-SW_T{}_lr{}_times_seed{}.txt".format(L,s_lr,seed), np.array(caltimes), delimiter=",")


for kappa in [1,10,50]:
    for s_lr in [0.1,0.01]:
        for L in [10]:
            for T in [10]:
                for seed in [1,2,3]:
                    points=[]
                    caltimes=[]
                    distances=[]
                    X=torch.tensor(source, requires_grad=True)
                    optimizer = torch.optim.SGD([X], lr=learning_rate)
                    start = time.time()
                    for i in range(N_step):
                        if (i in print_steps):
                            distance, cal_time = compute_true_Wasserstein(X, Y), time.time() - start
                            print("v-DSW {}:{} ({}s)".format(i+1,compute_true_Wasserstein(X,Y),np.round(time.time()-start,2)))
                            points.append(X.clone().data.numpy())
                            caltimes.append(cal_time)
                            distances.append(distance)
                        optimizer.zero_grad()
                        sw= N*vDSW(X,Y,kappa=kappa,s_lr=s_lr,n_lr=T,L=L)
                        sw.backward()
                        optimizer.step()
                    points.append(Y.clone().data.numpy())
                    np.save("saved/v-DSW_kappa{}_L{}_T{}_lr{}_points_seed{}.npy".format(kappa,L,T,s_lr,seed),np.stack(points))
                    np.savetxt("saved/v-DSW_kappa{}_L{}_T{}_lr{}_distances_seed{}.txt".format(kappa,L,T,s_lr,seed), np.array(distances), delimiter=",")
                    np.savetxt("saved/v-DSW_kappa{}_L{}_T{}_lr{}_times_seed{}.txt".format(kappa,L,T,s_lr,seed), np.array(caltimes), delimiter=",")


#
for L in [10,100]:
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True)
        points=[]
        caltimes=[]
        distances=[]
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance, cal_time = compute_true_Wasserstein(X, Y), time.time() - start
                print("ISEBSW {}:{} ({}s)".format(i + 1, compute_true_Wasserstein(X, Y), np.round(time.time() - start, 2)))
                points.append(X.clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            sw= N*ISEBSW(X,Y,L=1,T=L,f_type=f_type,eps=eps,copy=copy)
            sw.backward()
            optimizer.step()
        points.append(Y.clone().data.numpy())
        np.save("saved/ISEBSW_L{}_f{}_{}_{}_{}_points_seed{}.npy".format(L,f_type,copy,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/ISEBSW_L{}_f{}_{}_{}_{}_points_seed{}_distances.txt".format(L,f_type,copy,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/ISEBSW_L{}_f{}_{}_{}_{}_points_seed{}_times.txt".format(L,f_type,copy,ind1,ind2,seed), np.array(caltimes), delimiter=",")

#
#
for L in [10,100]:
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True)
        points=[]
        caltimes=[]
        distances=[]
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance, cal_time = compute_true_Wasserstein(X, Y), time.time() - start
                print("IMHEBSW {}:{} ({}s)".format(i + 1, compute_true_Wasserstein(X, Y), np.round(time.time() - start, 2)))
                points.append(X.clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)

            optimizer.zero_grad()
            sw= N*IMHEBSW(X,Y,L=1,T=L,f_type=f_type,copy=copy,eps=eps)
            sw.backward()
            optimizer.step()
        points.append(Y.clone().data.numpy())
        np.save("saved/IMHEBSW_L{}_f{}_{}_{}_{}_points_seed{}.npy".format(L, f_type, copy, ind1, ind2, seed),
                np.stack(points))
        np.savetxt(
            "saved/IMHEBSW_L{}_f{}_{}_{}_{}_points_seed{}_distances.txt".format(L, f_type, copy, ind1, ind2, seed),
            np.array(distances), delimiter=",")
        np.savetxt("saved/IMHEBSW_L{}_f{}_{}_{}_{}_points_seed{}_times.txt".format(L, f_type, copy, ind1, ind2, seed),
                   np.array(caltimes), delimiter=",")


for L in [10,100]:
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True)
        points=[]
        caltimes=[]
        distances=[]
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance, cal_time = compute_true_Wasserstein(X, Y), time.time() - start
                print("RMHEBSW {}:{} ({}s)".format(i + 1, compute_true_Wasserstein(X, Y), np.round(time.time() - start, 2)))
                points.append(X.clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)

            optimizer.zero_grad()
            sw= N*RMHEBSW(X,Y,L=1,T=L,kappa=10,f_type=f_type,copy=copy,eps=eps)
            sw.backward()
            optimizer.step()
        points.append(Y.clone().data.numpy())
        np.save("saved/RMHEBSW_L{}_f{}_{}_{}_{}_points_seed{}.npy".format(L, f_type, copy, ind1, ind2, seed),
                np.stack(points))
        np.savetxt(
            "saved/RMHEBSW_L{}_f{}_{}_{}_{}_points_seed{}_distances.txt".format(L, f_type, copy, ind1, ind2, seed),
            np.array(distances), delimiter=",")
        np.savetxt("saved/RMHEBSW_L{}_f{}_{}_{}_{}_points_seed{}_times.txt".format(L, f_type, copy, ind1, ind2, seed),
                   np.array(caltimes), delimiter=",")




for L in [10,100]:
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True)
        points=[]
        caltimes=[]
        distances=[]
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance, cal_time = compute_true_Wasserstein(X, Y), time.time() - start
                print("SIR {}:{} ({}s)".format(i + 1, compute_true_Wasserstein(X, Y), np.round(time.time() - start, 2)))
                points.append(X.clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            sw= N*SIREBSW(X,Y,L=1,T=L,f_type=f_type,copy=copy,eps=eps)
            sw.backward()
            optimizer.step()
        points.append(Y.clone().data.numpy())
        np.save("saved/SIREBSW_L{}_f{}_{}_{}_{}_points_seed{}.npy".format(L, f_type, copy, ind1, ind2, seed),
                np.stack(points))
        np.savetxt(
            "saved/SIREBSW_L{}_f{}_{}_{}_{}_points_seed{}_distances.txt".format(L, f_type, copy, ind1, ind2, seed),
            np.array(distances), delimiter=",")
        np.savetxt("saved/SIREBSW_L{}_f{}_{}_{}_{}_points_seed{}_times.txt".format(L, f_type, copy, ind1, ind2, seed),
                   np.array(caltimes), delimiter=",")