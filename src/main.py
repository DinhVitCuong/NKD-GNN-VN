import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from preprocess_image import resize_image
from model import image_captioning_model
from knowledge_graph import build_knowledge_graph
from template_caption import generate_template_caption