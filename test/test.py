import numpy as np
v=0.1
a=1/3.0 * v ** 3
# print(a)
# import torch
# torch.manual_seed(123)
# np.random.seed(123)
# for i in range(10):
#     print()
#     print(np.random.randint(0, 900))

# import torch.nn.functional as F
# from torch.distributions import Normal
# from itertools import zip_longest
# from typing import Dict, List, Tuple, Type, Union, Any
# from itertools import chain
# LOG_SIG_MAX = 0.05
# LOG_SIG_MIN = 0.025
import torch
from torch.distributions import  Normal
mean=torch.Tensor([-1,1])
normal=Normal(mean,0.5)
c=normal.sample()
print("c:",c)
# def encode(s):
#     return ' '.join([bin(ord(c)).replace('0b', '') for c in s])
 
# def decode(s):
#     return ''.join([chr(i) for i in [int(b, 2) for b in s.split(' ')]])
# a=256

    
# # print(bin(a).replace('0b', ''))
# b=bin(a).replace('0b', '')
# for i, s in enumerate(b):
#     print(i)
#     print(s)


'''
actor_init
    run5 not ia
    run6  cnn ia
    run7 not lt
pre_critic 
    run2 lt dead
    run3 ia dead
sac
    run3 lt dead
    run4 ia dead
'''