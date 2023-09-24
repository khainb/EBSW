import numpy as np


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def sample_pc(pc, num_points):
    choice = np.random.choice(pc.shape[0], num_points, replace=True)
    return pc[choice, :]


if __name__ == "__main__":
    opc = 15.0 * np.random.rand(2, 3)
    print(opc)
    print(pc_normalize(opc))
