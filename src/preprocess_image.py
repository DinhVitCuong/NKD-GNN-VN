from PIL import Image
from torchvision import transforms

def resize_image(image_dir):
    image = Image.open(image_dir).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match VGG19 input size
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image