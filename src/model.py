import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import from_networkx

class ImageEncoder(nn.Module):
    """
    Mô hình trích xuất đặc trưng ảnh sử dụng ResNet-50
    """
    def __init__(self, embed_size, pretrained=True):
        super(ImageEncoder, self).__init__()
        
        # Load ResNet-50 pre-trained
        self.cnn = models.resnet50(pretrained=pretrained)
        
        # Thay thế lớp fully connected cuối cùng
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, images):
        features = self.cnn(images)  # (batch_size, embed_size)
        return features

class CaptionDecoder(nn.Module):
    """
    Module tạo caption sử dụng LSTM với cơ chế attention
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features):
        batch_size = features.size(0)
        h = features.unsqueeze(0)
        c = torch.zeros_like(h)
        outputs = []
        input_word = torch.zeros(batch_size, dtype=torch.long).to(features.device)
        
        for _ in range(20):  # Giới hạn độ dài caption
            embeddings = self.fc(h[-1])
            out, (h, c) = self.lstm(embeddings.unsqueeze(1), (h, c))
            output = self.fc(out.squeeze(1))
            outputs.append(output.argmax(dim=1))
        
        return torch.stack(outputs, dim=1)

class ImageCaptionModel(nn.Module):
    """
    Mô hình end-to-end hoàn chỉnh (loại bỏ NKDGNN và bỏ input caption)
    """
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        
        # Các thành phần
        self.encoder = ImageEncoder(embed_size)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size)
        
    def forward(self, images):
        # Trích xuất đặc trưng
        visual_features = self.encoder(images)  # (batch_size, embed_size)
        
        # Tạo caption
        outputs = self.decoder(visual_features)
        
        return outputs

