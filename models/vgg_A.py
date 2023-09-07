import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_A(nn.Module):

    def __init__(self, num_classes = 10):
        super(VGG_A, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode= True)

        self.conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv5_1 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(32768, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1_1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool5(x)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x