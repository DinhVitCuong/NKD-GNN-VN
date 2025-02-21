import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torchvision.models as models

from preprocess_image import resize_image

######################## TEMPLATE CAPTION GENERATOR ###############################

# CNN Encoder with Pre-trained
class VGG19Encoder(nn.Module):
    def __init__(self, embed_size):
        super(VGG19Encoder, self).__init__()
        # Load pre-trained
        vgg = models.vgg19(pretrained=True)
        # Remove the last fully connected layer (fc) of 
        self.vgg = nn.Sequential(*list(vgg.children())[:-1])
        # Linear layer to map ResNet output to embedding size
        self.fc = nn.Linear(vgg.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Extract features using 
        with torch.no_grad():  # Disable gradient computation for ResNet
            features = self.vgg(images)
        # Flatten the features and pass through a linear layer
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.relu(features)
        features = self.dropout(features)
        return features
    
class LSTMDecoder(nn.Module):
    def __init__(self, embed_size = 256, hidden_size = 512, vocab_size =10000, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        return outputs
    
# Full model combining CNN encoder and LSTM decoder
class image_captioning_model(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.encoder = VGG19Encoder(embed_size)
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

################################ NKDGNN #####################################

class NKDGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(NKDGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # Initialize the GCN layers (using GCNConv to use edge weights)
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))
        
        # Attention mechanism parameters
        self.q = nn.Parameter(torch.Tensor(hidden_dim))
        self.W1 = nn.Linear(output_dim, hidden_dim)
        self.W2 = nn.Linear(output_dim, hidden_dim)
        self.W3 = nn.Linear(2 * hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # Apply GCN layers with edge weights and ReLU activation
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index, edge_weight=edge_weight))
        x = self.layers[-1](x, edge_index, edge_weight=edge_weight)
        
        # Generate the global representation vector Ng using attention
        global_vector = self.calculate_global_vector(x)
        key_vector = self.calculate_key_entity_vector(x, data.edge_index)
        
        # Compute the news knowledge graph representation Nr
        Nr = self.generate_graph_representation(global_vector, key_vector)
        return Nr

    def calculate_global_vector(self, x):
        """Calculates the global representation vector Ng using attention mechanism."""
        alpha = torch.tanh(self.W1(x) + self.W2(x.mean(dim=0)))
        alpha = torch.matmul(alpha, self.q)
        attention_weights = self.softmax(alpha)
        Ng = torch.sum(attention_weights.unsqueeze(-1) * x, dim=0)
        return Ng

    def calculate_key_entity_vector(self, x, edge_index):
        """Finds and returns the vector of the key entity (node with the most edges)."""
        degrees = torch.bincount(edge_index[0])
        key_entity_index = degrees.argmax()
        return x[key_entity_index]

    def generate_graph_representation(self, global_vector, key_vector):
        """Generates the final graph representation vector Nr."""
        combined_vector = torch.cat((global_vector, key_vector), dim=0)
        Nr = torch.tanh(self.W3(combined_vector))
        return Nr

class PredictionModule(nn.Module):
    def __init__(self, input_dim):
        super(PredictionModule, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, entity_vectors, graph_representation):
        z_hat = torch.matmul(entity_vectors, graph_representation.T)
        y_hat = self.softmax(z_hat)
        return y_hat

def loss_function(y_hat, y_true):
    """Computes the cross-entropy loss."""
    loss = -torch.sum(y_true * torch.log(y_hat) + (1 - y_true) * torch.log(1 - y_hat))
    return loss

def get_node_probabilities(model, prediction_module, data):
    """
    Given the NKDGNN model and PredictionModule, returns the probabilities for each node in the graph.
    
    Parameters:
    - model: The NKDGNN model instance
    - prediction_module: The PredictionModule instance
    - data: The graph data containing node features, edge indices, etc.
    
    Returns:
    - y_hat: The probabilities of each node being selected as the placeholder.
    """
    # Pass the data through NKDGNN to get the graph representation Nr
    Nr = model(data)  # Output from NKDGNN, representing the entire graph
    
    # Get the probabilities for each node using the PredictionModule
    y_hat = prediction_module(data.x, Nr)  # y_hat contains probabilities for each node
    
    return y_hat

########################## CAPTION GENERATOR ###########################

class CaptionGenerator:
    def __init__(self, model, prediction_module):
        self.model = model
        self.prediction_module = prediction_module

    def predict_caption(self, data, template):
        # Pass the data through NKDGNN to get the graph representation Nr
        Nr = self.model(data)
        
        # Get the probabilities for each node using the prediction module
        y_hat = self.prediction_module(data.x, Nr)  # y_hat contains probabilities for each node
        
        # Decode the predicted nodes based on the placeholders in the template
        filled_caption = self.fill_template_with_entities(template, data, y_hat)
        return filled_caption

    def fill_template_with_entities(self, template, data, y_hat):
        """
        Fills in the template with the most likely entities based on the node types and probabilities.
        """
        # Example placeholders and their corresponding node types in the graph
        placeholders = ['<PERSON>', '<ORGANIZATION>', '<PLACE>', '<BUILDING>']
        
        # Initialize a dictionary to store the best matching nodes for each placeholder
        entity_mapping = {placeholder: None for placeholder in placeholders}

        # Iterate over the placeholders
        for placeholder in placeholders:
            # Find nodes in the graph that match the type of the placeholder
            matching_indices = [
                i for i, node_type in enumerate(data.node_types) if node_type == placeholder
            ]
            
            if matching_indices:
                # Select the node with the highest probability among the matching nodes
                best_node_idx = max(matching_indices, key=lambda idx: y_hat[idx].item())
                entity_mapping[placeholder] = data.node_labels[best_node_idx]

        # Fill the template with the selected entities
        filled_caption = template
        for placeholder, entity in entity_mapping.items():
            if entity:
                filled_caption = filled_caption.replace(placeholder, entity)
        
        return filled_caption