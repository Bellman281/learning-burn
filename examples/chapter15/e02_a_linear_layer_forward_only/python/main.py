import torch

x = torch.tensor([[1., 2., 3.]])                     # [1, 3]
W = torch.tensor([[.1, .2], [.3, .4], [.5, .6]])     # [3, 2]
b = torch.tensor([.5, -.5])                          # [2]

y = torch.relu(x @ W + b)                            # [[2.7, 2.3]]

print(f"output shape = {list(y.shape)}")
print(f"output       = {y.tolist()}")
