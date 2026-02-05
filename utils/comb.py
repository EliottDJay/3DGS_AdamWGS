

import math
import torch
from scipy.special import comb

N_max = 51
binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n+1):
        binoms[n, k] = math.comb(n, k)

print(binoms)

binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n+1):
        binoms[n, k] = comb(n, k)  

print(binoms)

    