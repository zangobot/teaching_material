# Inspired by https://github.com/maurapintor/mnist-pretrained
# and by https://github.com/unica-ml/ml/blob/master/notebooks/lab06.ipynb
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNet(torch.nn.Module):
	def __init__(self):

		super(SimpleNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
		self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
		self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
		self.fc1 = nn.Linear(1024, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, 10)

	def forward(self, x):
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = torch.max_pool2d(x, 2)

		x = torch.relu(self.conv3(x))
		x = torch.relu(self.conv4(x))
		x = torch.max_pool2d(x, 2)

		x = x.view(-1, 1024)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)

	def train_on_mnist(self, name=None):
		transform = self.get_transform()
		mnist_dataset = datasets.MNIST(download=True, root='.', train=True, transform=transform)
		train_loader = DataLoader(mnist_dataset, batch_size=256, shuffle=True)

		optimizer = Adam(lr=0.01, params=self.parameters())
		loss_fn = torch.nn.CrossEntropyLoss()
		epochs = 2
		self.train()
		for e in range(epochs):
			print(f'EPOCH {e}')
			for i, (samples, labels) in enumerate(train_loader):
				optimizer.zero_grad()
				preds = self(samples)
				loss = loss_fn(preds, labels)
				loss.backward()
				optimizer.step()
				if i % 10 == 1:
					print(f'Batch {i}, train loss: {loss}')

		torch.save(self.state_dict(), name if name is not None else './mnist_net.pth')
		return self

	def load_pretrained_mnist(self, name=None):
		state_dict = torch.load(name if name is not None else './mnist_net.pth')
		self.load_state_dict(state_dict)
		self.eval()
		return self

	def get_transform(self):
		return transforms.ToTensor()

	def test_mnist(self):
		self.eval()
		accuracy = 0
		transform = self.get_transform()
		mnist_test_dataset = datasets.MNIST(download=True, root='.', train=False, transform=transform)
		test_loader = DataLoader(mnist_test_dataset, batch_size=256)
		for tdata in test_loader:
			samples, labels = tdata
			pred_labels = self(samples).argmax(dim=1, keepdim=True)
			accuracy += pred_labels.eq(labels.view_as(pred_labels)).sum().item()
		accuracy = accuracy / len(test_loader.dataset)
		print(f'Test accuracy: {accuracy}')
