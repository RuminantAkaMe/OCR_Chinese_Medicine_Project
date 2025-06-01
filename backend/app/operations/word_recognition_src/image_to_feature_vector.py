# processor/image_to_vector.py

import torch
from torchvision import models, transforms
from PIL import Image

# Load a pretrained ResNet18 model and convert it to half-precision (float16) for efficiency.
# Half-precision reduces memory usage and computation time, but may slightly reduce accuracy.
model = models.resnet18(pretrained=True).half()
model.eval()  # Set model to evaluation mode (disables dropout, etc.)

# Define preprocessing transformations:
# - Resize image to 224x224 (standard for ResNet)
# - Convert image to PyTorch tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Convert an image at the given path into a feature vector using ResNet18
def image_to_feature_vector(img_path):
    # Open and convert image to RGB
    img = Image.open(img_path).convert("RGB")
    
    # Apply preprocessing and add batch dimension
    tensor = transform(img).unsqueeze(0)
    
    # Convert tensor to half-precision to match model
    tensor = tensor.to(torch.float16)
    
    # Run inference without tracking gradients (saves memory)
    with torch.no_grad():
        embedding = model(tensor).squeeze(0)  # Remove batch dimension

    # Return embedding as a list (or numpy array if needed)
    return embedding.tolist()

