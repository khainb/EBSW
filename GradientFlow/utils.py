import numpy as np
import torch
import ot
from von_mises_fisher import VonMisesFisher
def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

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



def MaxSW(X,Y,p=2,s_lr=0.1,n_lr=100,device="cpu",adam=False):
    dim = X.size(1)
    theta = torch.randn((1, dim), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    if(adam):
        optimizer = torch.optim.Adam([theta], lr=s_lr)
    else:
        optimizer = torch.optim.SGD([theta], lr=s_lr)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(n_lr-1):
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_detach,Y_detach,theta,p=p).mean(),1./p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1,keepdim=True))
    sw = one_dimensional_Wasserstein_prod(X, Y,theta, p=p).mean()
    return torch.pow(sw,1./p)

def vDSW(X,Y,L=10,kappa=50,p=2,s_lr=0.1,n_lr=100,device="cpu"):
    dim = X.size(1)
    epsilon = torch.randn((1, dim), device=device, requires_grad=True)
    epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1,keepdim=True))
    optimizer = torch.optim.SGD([epsilon], lr=s_lr)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(n_lr-1):
        vmf = VonMisesFisher(epsilon, torch.full((1, 1), kappa, device=device))
        theta = vmf.rsample(L).view(L, -1)
        negative_sw = -torch.pow(one_dimensional_Wasserstein_prod(X_detach,Y_detach,theta,p=p).mean(),1./p)
        optimizer.zero_grad()
        negative_sw.backward()
        optimizer.step()
        epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1,keepdim=True))
    vmf = VonMisesFisher(epsilon, torch.full((1, 1), kappa, device=device))
    theta = vmf.rsample(L).view(L, -1)
    sw = one_dimensional_Wasserstein_prod(X, Y,theta, p=p).mean()
    return torch.pow(sw,1./p)

def SW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)



def ISEBSW(X, Y, L=1,T=10, p=2, f_type="poly",eps=0,copy=True, rho=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L*T,device)
    wasserstein_distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    wasserstein_distances =  wasserstein_distances.view(L,T)
    if(f_type=="exp"):
        weights = torch.softmax(wasserstein_distances,dim=1)
    elif(f_type=="identity"):
        weights =  wasserstein_distances + eps
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
    elif (f_type == "poly"):
        weights = wasserstein_distances**rho + eps
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
    if(copy):
        weights = weights.detach()
    sw = torch.sum(weights*wasserstein_distances,dim=1).mean()
    return  torch.pow(sw,1./p)


def SIREBSW(X, Y, L=1,T=10, p=2,f_type="poly",copy=True,eps=0,rho=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L * T, device)
    wasserstein_distances = one_dimensional_Wasserstein_prod(X, Y, theta, p=p)
    wasserstein_distances = wasserstein_distances.view(L, T)
    if (f_type == "exp"):
        weights = torch.softmax(wasserstein_distances, dim=1)
    elif (f_type == "identity"):
        weights = wasserstein_distances + eps
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
    elif (f_type == "poly"):
        weights = wasserstein_distances**rho + eps
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
    inds = torch.multinomial(weights, T, replacement=True)
    theta = theta.view(L,T,dim)
    thetas =[]
    for i in range(L):
        thetas.append(theta[i][inds[i]])
    theta = torch.cat(thetas,dim=0)
    if(copy):
        sw = one_dimensional_Wasserstein_prod(X, Y, theta, p=p).mean()
        return torch.pow(sw, 1. / p)
    else:
        wasserstein_distances = one_dimensional_Wasserstein_prod(X, Y, theta, p=p)
        wasserstein_distances = wasserstein_distances.view(L, T)
        wasserstein_distances_detach = wasserstein_distances.detach()

        if (f_type == "exp"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = torch.exp(wasserstein_distances)
            constant = torch.mean(torch.exp(distances), dim=1, keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                    wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()
            # theta_constant = rand_projections(dim, L * T, device)
            # distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            #
            # return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
            #             wasserstein_distances_detach *  (torch.log(wasserstein_distances) + wasserstein_distances -torch.logsumexp(distances,dim=1,keepdim=True) ) ).mean()
        elif (f_type == "identity"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = wasserstein_distances + eps
            constant = torch.mean(distances + eps, dim=1, keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                    wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()
        elif (f_type == "poly"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = wasserstein_distances ** rho
            constant = torch.mean(distances ** rho, dim=1, keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                    wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()

def IMHEBSW(X, Y, L=1,T=10,M=0,N=1, p=2,f_type="poly",copy=True,eps=0,rho=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L , device)
    thetas = [theta]
    Xdetach=X.detach()
    Ydetach=Y.detach()
    with torch.no_grad():
        for l in range(T-1):
            theta_new = rand_projections(dim, L, device)
            distance_new = one_dimensional_Wasserstein_prod(Xdetach, Ydetach,theta_new, p=p)
            theta_old = thetas[-1]
            distance_old = one_dimensional_Wasserstein_prod(Xdetach, Ydetach,theta_old, p=p)
            if (f_type == "exp"):
                log_ratio = distance_new-distance_old
            elif (f_type == "identity"):
                log_ratio = torch.log(distance_new+eps) - torch.log(distance_old+eps)
            elif (f_type == "poly"):
                log_ratio = torch.log(distance_new**rho+eps) -torch.log(distance_old**rho+eps)
            acceptance_rate = torch.min(torch.cat([torch.log(torch.ones(log_ratio.shape)),log_ratio],dim=1),dim=1,keepdim=True)[0]
            u = torch.log(torch.rand(acceptance_rate.shape))
            acceptance = (u<=acceptance_rate).int().repeat(1,dim)
            thetas.append(theta_new*acceptance +theta_old*(1-acceptance))
    theta = torch.cat(thetas[M:][::N],dim=0)
    if(copy):
        sw = one_dimensional_Wasserstein_prod(X, Y, theta, p=p).mean()
        return torch.pow(sw, 1. / p)
    else:
        wasserstein_distances = one_dimensional_Wasserstein_prod(X, Y, theta, p=p)
        wasserstein_distances = wasserstein_distances.view(L, T)
        wasserstein_distances_detach = wasserstein_distances.detach()

        if (f_type == "exp"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = torch.exp(wasserstein_distances)
            constant = torch.mean(torch.exp(distances) , dim=1, keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                    wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()
            # theta_constant = rand_projections(dim, L * T, device)
            # distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            #
            # return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
            #             wasserstein_distances_detach *  (torch.log(wasserstein_distances) + wasserstein_distances -torch.logsumexp(distances,dim=1,keepdim=True) ) ).mean()
        elif (f_type == "identity"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = wasserstein_distances + eps
            constant = torch.mean(distances+ eps,dim=1,keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                        wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()
        elif (f_type == "poly"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = wasserstein_distances**rho
            constant = torch.mean(distances**rho,dim=1,keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                        wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()

def RMHEBSW(X, Y, L=1,T=10,M=0,N=1,kappa=10, p=2,f_type="poly",copy=True,eps=0,rho=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L , device)
    thetas = [theta]
    Xdetach=X.detach()
    Ydetach=Y.detach()
    with torch.no_grad():
        for l in range(T-1):
            theta_old = thetas[-1]
            distance_old = one_dimensional_Wasserstein_prod(Xdetach, Ydetach,theta_old, p=p)
            vmf = VonMisesFisher(theta_old, torch.full((L, 1), kappa, device=device))
            theta_new = vmf.rsample(1).view(L, -1)
            distance_new = one_dimensional_Wasserstein_prod(Xdetach, Ydetach, theta_new, p=p)
            if (f_type == "exp"):
                log_ratio = distance_new - distance_old
            elif (f_type == "identity"):
                log_ratio = torch.log(distance_new + eps) - torch.log(distance_old + eps)
            elif (f_type == "poly"):
                log_ratio = torch.log(distance_new ** rho + eps) - torch.log(distance_old ** rho + eps)
            acceptance_rate = \
            torch.min(torch.cat([torch.log(torch.ones(log_ratio.shape)), log_ratio], dim=1), dim=1, keepdim=True)[0]
            u = torch.log(torch.rand(acceptance_rate.shape))
            acceptance = (u<=acceptance_rate).int().repeat(1,dim)
            thetas.append(theta_new*acceptance +theta_old*(1-acceptance))
    theta = torch.cat(thetas[M:][::N],dim=0)
    if(copy):
        sw = one_dimensional_Wasserstein_prod(X, Y, theta, p=p).mean()
        return torch.pow(sw, 1. / p)
    else:
        wasserstein_distances = one_dimensional_Wasserstein_prod(X, Y, theta, p=p)
        wasserstein_distances = wasserstein_distances.view(L, T)
        wasserstein_distances_detach = wasserstein_distances.detach()

        if (f_type == "exp"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = torch.exp(wasserstein_distances)
            constant = torch.mean(torch.exp(distances), dim=1, keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                    wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()
            # theta_constant = rand_projections(dim, L * T, device)
            # distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            #
            # return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
            #             wasserstein_distances_detach *  (torch.log(wasserstein_distances) + wasserstein_distances -torch.logsumexp(distances,dim=1,keepdim=True) ) ).mean()
        elif (f_type == "identity"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = wasserstein_distances + eps
            constant = torch.mean(distances + eps, dim=1, keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                    wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()
        elif (f_type == "poly"):
            theta_constant = rand_projections(dim, L * T, device)
            distances = one_dimensional_Wasserstein_prod(X, Y, theta_constant, p=p).view(L, T)
            f_weights = wasserstein_distances ** rho
            constant = torch.mean(distances ** rho, dim=1, keepdim=True)

            return 1. / p * torch.pow(wasserstein_distances_detach.mean(), (1. - p) / p) * (
                    wasserstein_distances_detach * torch.log(wasserstein_distances * f_weights / (constant))).mean()
