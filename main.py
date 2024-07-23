import torch

def f(arr):
    # arr  = arr + torch.ones(4,5)
    arr[:] = arr[:] + torch.ones(4,5)




t = torch.arange(1*2*3*4*5).reshape(1,2,3,4,5)
f(t)
print(t)