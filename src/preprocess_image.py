from PIL import Image
from torchvision import transforms

def resize_image(image_dir):
    image = Image.open(image_dir).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization values for ImageNet
                             std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image