from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch

#--------------LOAD THE DATA---------------------------------------
transform = transforms.Compose([transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				])

trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) 



#-----------------BULIDING THE NETWORK-----------------------------
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 10)

	def forward(self, x):
		x = x.view(x.shape[0], -1)
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.log_softmax(self.fc4(x), dim=1)

		return x

#------CREATE THE NETWORK, DEFINE THE CRITERION AND OPTIMIZER------
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
	
#----------------TRAIN THE NETWORK---------------------------------
epochs = 5

for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		logps = model(images)
		loss = criterion(logps, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
	else:
		print("Training loss: {}".format(running_loss))

