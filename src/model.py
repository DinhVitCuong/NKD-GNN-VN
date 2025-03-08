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

class NKDGNN(nn.Module):
    """
    News Knowledge-Driven Graph Neural Network (NKD-GNN)
    Dùng để phân tích quan hệ giữa các thực thể trong đồ thị tri thức và tính xác suất của các thực thể
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        
        # GNN Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        
        # Attention Mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Final Prediction Layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, data, entity_list):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT Layers
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        
        # Compute Global Graph Representation
        attention_weights = F.softmax(self.attention(x), dim=0)
        global_representation = torch.sum(attention_weights * x, dim=0)
        
        # Compute Entity Probabilities
        entity_scores = self.fc(x)
        entity_probs = torch.sigmoid(entity_scores).squeeze(-1)
        
        # Tạo danh sách thực thể với xác suất
        entity_prob_list = [(entity_list[i][0], entity_list[i][1], entity_probs[i].item()) for i in range(len(entity_list))]
        
        return entity_prob_list, global_representation