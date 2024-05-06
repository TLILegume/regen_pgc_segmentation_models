# Example of target with class indices
from torch import nn
import torch

loss = nn.CrossEntropyLoss()
input = torch.randn(2, 7, 1000, 1000, requires_grad=True)
# print(input)
target = torch.empty(2, 1000, 1000, dtype=torch.long).random_(7)

print(input.shape)
print(target.shape)
print(input.dtype)
print(target.dtype)

# print(input)
# print(target)
output = loss(input, target)
output.backward()
# Example of target with class probabilities
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)

# print(input)
# print(target)
# output = loss(input, target)
# output.backward()