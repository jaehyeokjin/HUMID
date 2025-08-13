# Python 3.7

import numpy as np

box_len = 19.73

with open('msd.coords', 'r') as f:
    coords = f.readlines()
coords = np.array(list(map(str.split, coords))).astype(float)
coords = coords[:, 2:5]

data = list()
for component in coords.T:
    component_ = list()
    for i in range(len(component)):
        if i % 10000 == 0:
            print(i)
        if i != 0:
            delta = component[i] - component[i - 1]
            if abs(delta) >= box_len / 2:
                component += -1 * np.sign(delta) * box_len
        component_.append(component[i])
    data.append(component_)

data = np.vstack(data)
np.save('unwrapped', data)
