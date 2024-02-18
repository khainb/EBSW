from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import torch
import time
torch.manual_seed(1)
np.random.seed(1)


from power_spherical import PowerSpherical
# p = PowerSpherical(
#       loc=torch.tensor([[0., 0,1.],[0., 1,0],[1., 0,0.] ], requires_grad=True),
#       scale=torch.tensor([4.,10.,50.], requires_grad=True),
#     )
# thetas=p.rsample((100,)).detach().cpu().numpy().reshape(-1,3)
# print(thetas.shape)
# # for L in [100]:
# #     torch.manual_seed(1)
# #     np.random.seed(1)
# #     thetas = torch.randn(L, 3)
# #     thetas = thetas / torch.sqrt(torch.sum(thetas ** 2, dim=1, keepdim=True))
# #     thetas = thetas.cpu().numpy()
# #     np.savetxt("MC"+str(L)+".csv", thetas, delimiter=",")
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = np.cos(u)*np.sin(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(v)
# ax.plot_wireframe(x, y, z, color="tab:grey",ls='--',alpha=0.5)
# for _ in range(1000):
#     a = torch.rand(100, 100)
# ax.scatter(thetas[:, 0], thetas[:, 1], thetas[:, 2])
# ax.set_title('Random')
# plt.show()
p = PowerSpherical(
      loc=torch.tensor([[0., 0,1.]], requires_grad=True),
      scale=torch.tensor([4.], requires_grad=True),
    )
thetas=p.rsample((100,)).view(100,-1)
print(torch.sum(thetas**2,dim=1))
print(thetas.shape)