import torch

x = torch.tensor([[1., 2., 3.]])                     # [1, 3]

# Layer 1: 3 -> 2, then ReLU.
W1 = torch.tensor([[.1, .2], [.3, .4], [.5, .6]])
b1 = torch.tensor([0., 0.])

h      = torch.relu(x @ W1 + b1)                     # [1, 2]

# Layer 2: 2 -> 3 (raw logits for 3 classes).
W2     = torch.tensor([[1., -1., .5], [.5, 1., -.5]])
logits = h @ W2                                      # [1, 3]

probs  = torch.softmax(logits, dim=1)                # built in

print(f"logits = {logits.tolist()}")
print(f"probs  = {probs.tolist()}")
print(f"prob sum = {probs.sum(dim=1, keepdim=True).tolist()}")
