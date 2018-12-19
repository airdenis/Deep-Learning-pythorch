from torch import nn
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				])

trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) 

dataiter = iter(trainloader)


model = nn.Sequential(nn.Linear(784, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 10),
			nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))

images = images.view(images.shape[0], -1)
logits = model(images)
loss = criterion(logits, labels)


print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad)

#-----------------------------------------------------------------------------------
from torch import optim

model = nn.Sequential(nn.Linear(784, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 10),
			nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		images = images.view(images.shape[0], -1)
		optimizer.zero_grad()

		output = model.forward(images)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
	else:
		print("Training loss:{}".format(running_loss/len(trainloader)))
