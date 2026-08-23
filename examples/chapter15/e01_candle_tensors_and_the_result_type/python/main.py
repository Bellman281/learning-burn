import torch

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])   # [2, 3]
b = torch.tensor([[1., 1., 1.], [2., 2., 2.]])

s  = a + b
p  = a * b
c  = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
mm = a @ c                                       # [2, 2]

# no Result, no ?; errors are exceptions
print(f"a shape = {list(a.shape)}")
print(f"a + b   = {s.tolist()}")
print(f"a * b   = {p.tolist()}")
print(f"a @ c   = {mm.tolist()}")
