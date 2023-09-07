import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VGG


BATCH_SIZE = 4
EPOCHS = 2


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.Resize(256)])
trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
valset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)


model = VGG()


criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


for epoch in range(EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')
print('Finished Training')