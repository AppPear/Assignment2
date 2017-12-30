from __future__ import print_function
import torch

'''What is PyTorch?'''

x = torch.rand(5,3) #rnadomly init tensor

# X:
print(x)

#  Size:
print(x.size())

y = torch.rand(5,3)

# Operations on tensors
print(x+y)
print(torch.add(x,y))

# Direct result to output:
result = torch.rand(3,5)
torch.add(x,y,out=result)
print(result)

# Inline operation
# Any operation that mutates a tensor in-place is post-fixed with an _ For example: x.copy_(y), x.t_(), will change x.
y.add_(x)
print(y)

# Numpy indexing:
print(result[:,1])
print(result[1,:])


'''Autograd: automatic differentiation'''

from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad = True)
print(x)

y = x+2
print(y)
print(y.grad_fn)
z = y*y*3
out = z.mean()
print(z,out)

out.backward()
print(x.grad)