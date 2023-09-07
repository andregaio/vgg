import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_dataset(dataset = 'cifar10', batch_size = 16):

    if dataset == 'cifar10':

        train_transform = transforms.Compose([transforms.ToTensor(),
            transforms.RandomResizedCrop(256, (0.08, 1), (3 / 4, 4 / 3)),
            transforms.RandomCrop(256),
            transforms.ColorJitter(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        val_transform = transforms.Compose([transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
        ])
        

        trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
        valset = datasets.CIFAR10(root='data', train=False, download=True, transform=val_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader