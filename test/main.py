import torch
import os
import torch.nn.functional as F
from torchvision import transforms
from models import ResNetWithDropout, ResNetBlockWithDropout
from optimizers import AdaIPS_S
from PIL import Image


root_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(root_dir, 'checkpoints', 'adaips_cifar_vanilla.pth')

print(os.path.abspath(__file__))

try:
    model = torch.load(checkpoint_path, weights_only=False)
except Exception as e:
    print('failed loading')
   


model.eval()

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 mean/std
])

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

image_path = os.path.join(root_dir, 'images', 'cat.jpg')
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    outputs = model(image)
    predicted_class = torch.argmax(outputs, dim=1).item()
print(f"Predicted Class: {classes[predicted_class]} ({predicted_class})")
print(f"confidence: {F.softmax(outputs, dim=1)[0][predicted_class]:10f}")