import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import logging
import py_vncorenlp

from preprocess_image import resize_image
from model import image_captioning_model, NKDGNN, PredictionModule, CaptionGenerator, get_node_probabilities, loss_function
from knowledge_graph import build_knowledge_graph,convert_networkx_to_data
from template_caption import generate_template_caption


logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

################## TRAINING #########################
def train(model, prediction_module, data_loader, optimizer, criterion, epochs):
    model.train()
    prediction_module.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for data in progress_bar:
            try:
                optimizer.zero_grad()
                Nr = model(data)  # Graph-level representation
                y_hat = prediction_module(data.x, Nr)  # Node-level predictions
                loss = criterion(y_hat, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})
            except Exception as e:
                logging.error(f"Error during training: {e}")
                progress_bar.set_postfix({"Error": "Check logs"})
        logging.info(f"Epoch {epoch + 1} completed with total loss: {total_loss:.4f}")
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")




##################### PREDICT ########################
def evaluate(model, prediction_module, data_loader, caption_generator):
    model.eval()
    prediction_module.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            try:
                Nr = model(data)
                node_probabilities = prediction_module(data.x, Nr)
                logging.info(f"Node Probabilities: {node_probabilities}")
                # Generate captions
                template = "<PERSON> met <ORGANIZATION> at <PLACE>."
                generated_caption = caption_generator.predict_caption(data, template)
                logging.info(f"Generated Caption: {generated_caption}")
                print("Generated Caption:", generated_caption)
            except Exception as e:
                logging.error(f"Error during evaluation: {e}")




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
    # Example parameter
    input_dim = 1
    hidden_dim = 64
    output_dim = 32
    num_layers = 3

    # Instantiate the model
    # Automatically download VnCoreNLP components from the original repository
    py_vncorenlp.download_model(save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

    # Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models`
    model_vncore = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner"], save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

    model = NKDGNN(input_dim, hidden_dim, output_dim, num_layers)
    prediction_module = PredictionModule(output_dim)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(prediction_module.parameters()), lr=0.01)
    criterion = loss_function
    caption_generator = CaptionGenerator(model, prediction_module)

    
    if args.isEval:
        logging.info("Starting evaluation...")
        print("Evaluating...")
        evaluate(model, prediction_module, data_loader, caption_generator)
    else:
        logging.info("Starting training...")
        print("Training...")
        train(model, prediction_module, data_loader, optimizer, criterion, args.epochs)