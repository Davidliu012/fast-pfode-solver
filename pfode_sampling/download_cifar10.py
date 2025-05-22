# save_real_cifar10.py
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

output_dir = "real_cifar10"
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
for i in range(len(dataset)):
    img, _ = dataset[i]
    save_image(img, os.path.join(output_dir, f"{i:05d}.png"))