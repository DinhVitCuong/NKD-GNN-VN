import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models

from preprocess_image import resize_image

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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
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