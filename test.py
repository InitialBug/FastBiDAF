import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


def max_k(p, length,beam_size):
    beam_size=min(length,beam_size)
    max_k_indices = [-1] * beam_size
    max_k_values = [-1] * beam_size
    for i in range(length):
        for j in range(beam_size):
            if p[i] > max_k_values[j]:
                if j == beam_size - 1:
                    max_k_values[j] = p[i]
                    max_k_indices[j] = i
                elif j==0:
                    max_k_values = [p[i]] + max_k_values[j:-1]
                    max_k_indices = [i] + max_k_indices[j:-1]
                else:
                    max_k_values = max_k_values[:j] + [p[i]] + max_k_values[j:-1]
                    max_k_indices = max_k_indices[:j] + [i] + max_k_indices[j:-1]
                break

    return max_k_indices

def beam_search(p1s, p2s,lengths,beam_size=3):
    a1 = []
    a2 = []

    for i in range(p1s.shape[0]):
        p1 = p1s[i]
        p2 = p2s[i]
        max_p1=max_k(p1,lengths[i],beam_size)
        max_p2=max_k(p2,lengths[i],beam_size)
        max=-1
        m1,m2=-1,-1
        for index1 in max_p1:
            for index2 in max_p2:
                if index1<=index2 and p1[index1]*p2[index2]>max:
                    max=p1[index1]*p2[index2]
                    m1=index1
                    m2=index2
        a1.append(m1)
        a2.append(m2)

    return a1, a2

# lr=0.001
# model=nn.Linear(1,1)
# optimizer = optim.Adam(betas=(0.8, 0.999), eps=1e-7, params=model.parameters())
# crit = lr / math.log2(1000)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: 0.1*ee)
#
# for i in range(10):
#     if i>0:
#         for param_group in optimizer.param_groups:
#             print(param_group['lr'])
#     scheduler.step()
# a=torch.Tensor([[[1,2,3,4,5,6],[5,6,7,8,9,0]]])
# c,d=torch.split(a,3,2)
# print(c)
# print(d)
a=2**3
print(a)


