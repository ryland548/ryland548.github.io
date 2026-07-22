from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

trainData = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
testData = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

loaders = {
    'train': DataLoader(trainData, batch_size=100, shuffle=True, num_workers=0),
    'test': DataLoader(testData, batch_size=100, shuffle=True, num_workers=0)
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2Drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2Drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: mps")
else:
    device = torch.device("cpu")
    print("Device: cpu")

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
lossFn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batchIdx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossFn(output, target)
        loss.backward()
        optimizer.step()
        if batchIdx % 25 == 0:
            print(f'epoch: {epoch}, batch: {batchIdx}, loss: {loss.item():.4f}')

def test():
    model.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += lossFn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(loaders["test"])
    print(f'\nTest set: Average Loss: {testLoss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.2f}%)\n')

MODEL_PATH = 'mnist_model.pth'
ONNX_PATH = 'mnist_model.onnx'

if os.path.exists(MODEL_PATH):
    print("Found saved weights — loading instead of retraining")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
else:
    print("No saved weights found — training from scratch")
    for epoch in range(1, 11):
        train(epoch)
        test()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved weights to {MODEL_PATH}")

model.eval().to("cpu")
dummy_input = torch.randn(1, 1, 28, 28, device="cpu")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=17
)
print(f"Exported to {ONNX_PATH}")
