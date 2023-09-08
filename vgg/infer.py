import torch
from models.base import VGG
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


CLASSES = 10
NETWORK = 'vgg_A'
WEIGHTS_FILEPATH  = 'weights/checkpoint_00070.pt'
CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


model = VGG(classes = CLASSES,  network = NETWORK)
model.load_state_dict(torch.load(WEIGHTS_FILEPATH))
model.eval()


val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, antialias=True),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


img = Image.open('assets/ship.png')
img = val_transform(img)
max_value, max_index = torch.max(F.softmax(model(img.unsqueeze(dim=0)), dim = 1), dim=1)
label = CLASS_NAMES[max_index.item()]
score = max_value.item()
print(f'Label: {label}; Score: {score:.2}')


