import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_dataset(dataset = 'cifar10', batch_size = 16):

    if dataset == 'cifar10':

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, (0.08, 1), (1, 1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
        ])
        

        trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
        valset = datasets.CIFAR10(root='data', train=False, download=True, transform=val_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)
        

    return trainloader, valloader