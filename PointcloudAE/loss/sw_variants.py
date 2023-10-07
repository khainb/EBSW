import os.path as osp
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

from .von_mises_fisher import VonMisesFisher

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def minibatch_rand_projections(batchsize, dim, num_projections=1000, device='cuda', **kwargs):
    projections = torch.randn((batchsize, num_projections, dim), device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=2, keepdim=True))
    return projections


def proj_onto_unit_sphere(vectors, dim=2):
    """
    input: vectors: [batchsize, num_projs, dim]
    """
    return vectors / torch.sqrt(torch.sum(vectors ** 2, dim=dim, keepdim=True))


def _sample_minibatch_orthogonal_projections(batch_size, dim, num_projections, device='cuda'):
    projections = torch.zeros((batch_size, num_projections, dim), device=device)
    projections = torch.stack([torch.nn.init.orthogonal_(projections[i]) for i in range(projections.shape[0])], dim=0)
    return projections


def compute_practical_moments_sw(x, y, num_projections=30, device="cuda", degree=2.0, **kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    dim = x.size(2)
    batch_size = x.size(0)
    projections = minibatch_rand_projections(batch_size, dim, num_projections, device=device)
    # projs.shape: [batchsize, num_projs, dim]

    xproj = x.bmm(projections.transpose(1, 2))

    yproj = y.bmm(projections.transpose(1, 2))

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = torch.pow(_sort_pow_p_get_sum.mean(dim=1), 1. / degree)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def compute_practical_moments_sw_with_predefined_projections(x, y, projections, device="cuda", degree=2.0, **kwargs):
    """
    x, y: [batch size, num points, dim]
    projections: [batch size, num projs, dim]
    """
    xproj = x.bmm(projections.transpose(1, 2))

    yproj = y.bmm(projections.transpose(1, 2))

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = torch.pow(_sort_pow_p_get_sum.mean(dim=1), 1. / degree)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment



class SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        squared_sw_2, _ = compute_practical_moments_sw(x, y, num_projections=self.num_projs, device=self.device)
        return {"loss": squared_sw_2.mean(dim=0)}


def compute_projected_distances(x, y, projections, degree=2.0, **kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    # projs.shape: [batchsize, num_projs, dim]

    xproj = x.bmm(projections.transpose(1, 2))

    yproj = y.bmm(projections.transpose(1, 2))

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    return _sort_pow_p_get_sum


def projx(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def expmap(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    EPS = {torch.float32: 1e-4, torch.float64: 1e-7}
    norm_u = u.norm(dim=-1, keepdim=True)
    exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
    retr = projx(x + u)
    cond = norm_u > EPS[norm_u.dtype]
    return torch.where(cond, exp, retr)


def proju(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    u = u - (x * u).sum(dim=-1, keepdim=True) * x
    return u


class EBSW(nn.Module):
    def __init__(self, L, device="cuda", **kwargs):
        super().__init__()
        self.L = L
        self.device = device
        self.f_type = kwargs["f_type"]
        self.p = kwargs["p"]
        self.T = kwargs["T"]
        self.eps = kwargs["eps"]
        self.kappa = kwargs["kappa"]
        self.estimation_type = kwargs["estimation_type"]
        # self.s_lr = kwargs["s_lr"]
        self.gradient_type = kwargs["gradient_type"]
        self.rho = kwargs["rho"]
        self.M = kwargs["M"]
        self.N = kwargs["N"]
        self.s_lr = kwargs["max_sw_lr"]

    def forward(self, x, y, **kwargs):
        dim = x.size(2)
        batch_size = x.size(0)

        if (self.estimation_type == "IS" or self.estimation_type == "SIR"):
            projections = minibatch_rand_projections(batch_size, dim, self.L, device=self.device)
            distances = compute_projected_distances(x, y, projections)
            if (self.f_type == "identity"):
                distances = distances + self.eps
                weights = distances / torch.sum(distances, dim=1, keepdim=True)
            elif (self.f_type == "exp"):
                weights = torch.softmax(distances, dim=1)
            if (self.estimation_type == "IS"):
                if (self.gradient_type == "independent"):
                    weights = weights.detach()
                return {"loss": (torch.pow(torch.sum(weights * distances, dim=1), 1. / self.p)).mean(dim=0)}
            elif (self.estimation_type == "SIR"):
                inds = torch.multinomial(weights, self.L, replacement=True)
                thetas = []
                for i in range(batch_size):
                    thetas.append(projections[i][inds[i]])
                projections = torch.stack(thetas, dim=0)
                distances = compute_projected_distances(x, y, projections)
                if (self.gradient_type == "independent"):
                    return {"loss": torch.pow(distances.mean(dim=1), 1. / self.p).mean(dim=0)}
                else:
                    distances_detach = distances.detach()
                    constant_projections = minibatch_rand_projections(batch_size, dim, self.L, device=self.device)
                    constant_distances = compute_projected_distances(x, y, projections)
                    if (self.f_type == "identity"):
                        f_distances = distances + self.eps
                        constant = torch.mean(constant_distances + self.eps, dim=1, keepdim=True)
                    elif (self.f_type == "exp"):
                        f_distances = torch.exp(distances)
                        constant = torch.mean(torch.exp(constant_distances), dim=1, keepdim=True)
                    p = self.p
                    loss = 1. / p * torch.pow(distances_detach.mean(dim=1), (1. - p) / p) * \
                           (distances_detach * torch.log(distances * f_distances / (constant))).mean(dim=1)
                    return {"loss": loss.mean()}
        elif (self.estimation_type == "IMH"):
            projections = minibatch_rand_projections(batch_size, dim, 1, device=self.device)
            thetas = [projections]
            with torch.no_grad():
                for l in range(self.L - 1):
                    theta_new = minibatch_rand_projections(batch_size, dim, 1, device=self.device)
                    distance_new = compute_projected_distances(x, y, theta_new)
                    theta_old = thetas[-1]
                    distance_old = compute_projected_distances(x, y, theta_old)
                    if (self.f_type == "exp"):
                        log_ratio = distance_new - distance_old
                    elif (self.f_type == "identity"):
                        log_ratio = torch.log(distance_new + self.eps) - torch.log(distance_old + self.eps)
                    acceptance_rate = \
                    torch.min(torch.cat([torch.zeros(log_ratio.shape, device=self.device), log_ratio], dim=1), dim=1,
                              keepdim=True)[0]
                    u = torch.log(torch.rand(acceptance_rate.shape, device=self.device))
                    acceptance = (u <= acceptance_rate).int().repeat(1, dim).float().view(theta_new.shape)
                    thetas.append(theta_new * acceptance + theta_old * (1 - acceptance))
            theta = torch.cat(thetas, dim=1)
            distances = compute_projected_distances(x, y, theta)
            if (self.gradient_type == "independent"):
                return {"loss": torch.pow(distances.mean(dim=1), 1. / self.p).mean(dim=0)}
            else:
                distances_detach = distances.detach()
                constant_projections = minibatch_rand_projections(batch_size, dim, self.L, device=self.device)
                constant_distances = compute_projected_distances(x, y, projections)
                if (self.f_type == "identity"):
                    f_distances = distances + self.eps
                    constant = torch.mean(constant_distances + self.eps, dim=1, keepdim=True)
                elif (self.f_type == "exp"):
                    f_distances = torch.exp(distances)
                    constant = torch.mean(torch.exp(constant_distances), dim=1, keepdim=True)
                p = self.p
                loss = 1. / p * torch.pow(distances_detach.mean(dim=1), (1. - p) / p) * \
                       (distances_detach * torch.log(distances * f_distances / (constant))).mean(dim=1)
                return {"loss": loss.mean()}
        elif (self.estimation_type == "RMH"):
            projections = minibatch_rand_projections(batch_size, dim, 1, device=self.device)
            thetas = [projections]
            with torch.no_grad():
                for l in range(self.L - 1):
                    theta_old = thetas[-1]
                    distance_old = compute_projected_distances(x, y, theta_old)
                    vmf = VonMisesFisher(theta_old.view(batch_size, dim),
                                         torch.full((batch_size, 1), self.kappa, device=self.device))
                    theta_new = vmf.rsample(1).view(batch_size, 1, dim)
                    distance_new = compute_projected_distances(x, y, theta_new)
                    if (self.f_type == "exp"):
                        log_ratio = distance_new - distance_old
                    elif (self.f_type == "identity"):
                        log_ratio = torch.log(distance_new + self.eps) - torch.log(distance_old + self.eps)
                    acceptance_rate = \
                    torch.min(torch.cat([torch.zeros(log_ratio.shape, device=self.device), log_ratio], dim=1), dim=1,
                              keepdim=True)[0]
                    u = torch.log(torch.rand(acceptance_rate.shape, device=self.device))
                    acceptance = (u <= acceptance_rate).int().repeat(1, dim).float().view(theta_new.shape)
                    thetas.append(theta_new * acceptance + theta_old * (1 - acceptance))
            theta = torch.cat(thetas, dim=1)
            distances = compute_projected_distances(x, y, theta)
            if (self.gradient_type == "independent"):
                return {"loss": torch.pow(distances.mean(dim=1), 1. / self.p).mean(dim=0)}
            else:
                distances_detach = distances.detach()
                constant_projections = minibatch_rand_projections(batch_size, dim, self.L, device=self.device)
                constant_distances = compute_projected_distances(x, y, projections)
                if (self.f_type == "identity"):
                    f_distances = distances + self.eps
                    constant = torch.mean(constant_distances + self.eps, dim=1, keepdim=True)
                elif (self.f_type == "exp"):
                    f_distances = torch.exp(distances)
                    constant = torch.mean(torch.exp(constant_distances), dim=1, keepdim=True)
                p = self.p
                loss = 1. / p * torch.pow(distances_detach.mean(dim=1), (1. - p) / p) * \
                       (distances_detach * torch.log(distances * f_distances / (constant))).mean(dim=1)
                return {"loss": loss.mean()}


class MaxSW(nn.Module):
    """
    Max-SW distance was proposed in paper "Max-Sliced Wasserstein Distance and its use for GANs" - CVPR'19
    The way to estimate it was proposed in paper "Generalized Sliced Wasserstein Distance" - NeurIPS'19
    """

    def __init__(self, device="cuda", **kwargs):
        super().__init__()
        self.device = device

    def forward(self, x, y, *args, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        dim = x.size(2)
        projections = Variable(
            minibatch_rand_projections(batchsize=x.size(0), dim=dim, num_projections=1, device=self.device),
            requires_grad=True,
        )
        # projs.shape: [batchsize, num_projs, dim]

        if kwargs["max_sw_optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                [projections],
                lr=kwargs["max_sw_lr"])
        elif kwargs["max_sw_optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                [projections],
                lr=kwargs["max_sw_lr"],
            )
        else:
            raise Exception("Optimizer has had implementation yet.")

        if kwargs["detach"]:
            x_detach, y_detach = x.detach(), y.detach()
        else:
            x_detach, y_detach = x, y

        for _ in range(kwargs["max_sw_num_iters"]):
            # compute loss
            xproj = x_detach.bmm(projections.transpose(1, 2))

            yproj = y_detach.bmm(projections.transpose(1, 2))

            _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

            _sort_pow_2_get_sum = torch.sum(torch.pow(_sort, 2), dim=2)

            if kwargs["squared_loss"]:
                negative_first_moment = -_sort_pow_2_get_sum.mean(dim=1)
            else:
                negative_first_moment = -torch.sqrt(_sort_pow_2_get_sum.mean(dim=1))

            # perform optimization
            optimizer.zero_grad()
            negative_first_moment.mean().backward()
            optimizer.step()
            # project onto unit sphere
            projections.data = proj_onto_unit_sphere(projections.data)

        projections_no_grad = projections.detach()
        loss, _ = compute_practical_moments_sw_with_predefined_projections(x, y, projections_no_grad,
                                                                           device=self.device)
        loss = loss.mean(dim=0)

        return {"loss": loss, "proj": projections_no_grad}



class VSW(nn.Module):
    """
    VSW - von Mises-Fisher Sliced Wasserstein
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, *args, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        batch_size, _, dim = x.size()
        locs = Variable(
            minibatch_rand_projections(batchsize=batch_size, dim=dim, num_projections=1, device=self.device).squeeze(1),
            requires_grad=True,
        )
        scales = torch.full((batch_size, 1), kwargs["kappa"], device=self.device)
        # projs.shape: [batchsize, num_projs, dim]

        if kwargs["max_sw_optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                [locs],
                lr=kwargs["max_sw_lr"])
        elif kwargs["max_sw_optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                [locs],
                lr=kwargs["max_sw_lr"],
            )
        else:
            raise Exception("Optimizer has had implementation yet.")

        if kwargs["detach"]:
            x_detach, y_detach = x.detach(), y.detach()
        else:
            x_detach, y_detach = x, y

        for _ in range(kwargs["max_sw_num_iters"]):
            # compute loss
            vmf = VonMisesFisher(locs, scales)
            projections = vmf.rsample(self.num_projs).transpose(0, 1)
            first_moment, _ = compute_practical_moments_sw_with_predefined_projections(x_detach, y_detach, projections,
                                                                                       device=self.device)
            if kwargs["squared_loss"]:
                negative_first_moment = -first_moment.mean()
            else:
                negative_first_moment = -torch.sqrt(first_moment).mean()

            # perform optimization
            optimizer.zero_grad()
            negative_first_moment.backward()
            optimizer.step()
            # project onto unit sphere
            locs.data = proj_onto_unit_sphere(locs.data, dim=1)

        locs_no_grad = locs.detach()
        vmf = VonMisesFisher(locs_no_grad, scales)
        # sample: num_projs x batch_size x dim
        # projections: batch_size x num_projs x dim
        projections = vmf.rsample(self.num_projs).transpose(0, 1)
        loss, _ = compute_practical_moments_sw_with_predefined_projections(x, y, projections, device=self.device)
        loss = loss.mean(dim=0)

        return {"loss": loss, "loc": locs_no_grad}




