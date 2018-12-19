import torch


def activation(x):
	return 1/(1+torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1, 2))

n_input = features.shape[1]
print n_input, 'n_input'
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)
print W1, 'W1'
W2 = torch.randn(n_input, n_output)
print W2, 'W2'

B1 = torch.randn((1, n_hidden))
print B1, 'B1'

B2 = torch.randn((1, n_output))
print B2, 'B2'

h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
print output
