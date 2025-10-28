import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# --- データ ---
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=True, download=True, transform=transform),
	batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=False, transform=transform),
	batch_size=1000, shuffle=False
)

# --- モデル ---
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28 * 28, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

# --- 学習ループ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 20):
	model.train()
	for data, target in train_loader:
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()
	print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# --- テスト ---
model.eval()
correct = 0
with torch.no_grad():
	for data, target in test_loader:
		data, target = data.to(device), target.to(device)
		output = model(data)
		pred = output.argmax(dim=1)
		correct += pred.eq(target).sum().item()

print(f"Test Accuracy: {correct / len(test_loader.dataset) * 100:.2f}%")