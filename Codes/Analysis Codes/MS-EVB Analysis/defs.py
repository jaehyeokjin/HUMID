import numpy as np


def msd(i, dt, coords):
    return np.linalg.norm(coords[i + dt] - coords[i]) ** 2
