# Python 3.7

import sys
import numpy as np
import multiprocessing as mp

from functools import partial
from defs import msd


if __name__ == "__main__":
    arr = np.load('unwrapped.npy').T
    dt_list = range(1, 25000)
    data = list()
    with mp.Pool() as pool:
        for dt in dt_list:
            print(dt)
            sys.stdout.flush()
            worker = partial(msd, dt=dt, coords=arr)
            _ = pool.map(worker, range(len(arr) - dt))
            data.append(np.average(_))
            if dt % 1000 == 0:
                np.save('msd', data)











