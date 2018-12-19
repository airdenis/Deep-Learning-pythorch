import torch

def activation(x):
	return 1/(1+torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1, 5))
print features

weights = torch.randn_like(features)
print weights

bias = torch.randn((1, 1))
print bias


multip = torch.mm(features, weights.view(5, 1))
print multip

y = activation(multip + bias)
print y
