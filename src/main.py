import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse

from preprocess_image import resize_image
from model import image_captioning_model, NKDGNN, PredictionModule, CaptionGenerator, get_node_probabilities, loss_function
from knowledge_graph import build_knowledge_graph,convert_networkx_to_data
from template_caption import generate_template_caption



# Training function
def train(model, prediction_module, data_loader, optimizer):
    model.train()
    prediction_module.train()
    for data in data_loader:
        optimizer.zero_grad()
        Nr = model(data)
        y_hat = prediction_module(data.x, Nr)
        loss = loss_function(y_hat, data.y)
        loss.backward()
        optimizer.step()

def predict(model, prediction_module, data, template):
    node_probabilities = get_node_probabilities(model, prediction_module, data)
    print("Node Probabilities:", node_probabilities)
    generator = CaptionGenerator(model, prediction_module)
    generated_caption = generator.predict_caption(data, template)
    print("Generated Caption:", generated_caption)

def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")
    # Argument for determine train or test
    parser.add_argument('--isEval', type=bool, default=True,
                        help='True: Evaluate, False: Train (Default True)')
 

    # Argument for the data path (train or test)
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to the dataset (.json files)')

    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training/testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training (default: 10)')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of layers (default: 3)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(f"Is Evaluate: {args.isEval}")
    print(f"Data Path: {args.datapath}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Number of Layers: {args.layers}")
    # Example usage
    input_dim = 1
    hidden_dim = 64
    output_dim = 32
    num_layers = 3

    # Instantiate the model
    model = NKDGNN(input_dim, hidden_dim, output_dim, num_layers)
    prediction_module = PredictionModule(output_dim)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(prediction_module.parameters()), lr=0.01)
    if args.isEval == True:
        print(f"Evaluating")
    else:
        print(f"Training")