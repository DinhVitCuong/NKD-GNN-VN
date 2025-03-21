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
    def __init__(self, embed_size, hidden_size, start_token, end_token, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.start_token = start_token
        self.end_token = end_token
        
    def forward(self, features, captions=None):
        batch_size = features.size(0)
        h = features.unsqueeze(0)  # (num_layers, batch_size, hidden_size)
        c = torch.zeros_like(h)
        
        # Training Mode: Sử dụng teacher forcing
        if captions is not None:
            embeddings = self.embed(captions)  # (batch_size, seq_len, embed_size)
            lstm_out, _ = self.lstm(embeddings, (h, c))  # (batch_size, seq_len, hidden_size)
            outputs = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
            return outputs
        
        # Inference Mode: Tự sinh từng từ
        else:
            input_words = torch.full((batch_size,), self.start_token, device=features.device)
            outputs = []
            for _ in range(20):  # Giới hạn độ dài tối đa
                embeddings = self.embed(input_words).unsqueeze(1)  # (batch_size, 1, embed_size)
                lstm_out, (h, c) = self.lstm(embeddings, (h, c))
                preds = self.fc(lstm_out.squeeze(1))  # (batch_size, vocab_size)
                next_words = preds.argmax(dim=1)
                outputs.append(next_words)
                input_words = next_words
                next_words = preds.argmax(dim=1)
                outputs.append(next_words)
                input_words = next_words
                # Dừng nếu gặp token <end>
                if (next_words == self.end_token).all():
                    break
            return torch.stack(outputs, dim=1)  # (batch_size,

class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = ImageEncoder(embed_size)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size)  # Truyền vocab_size vào đây
        
    def forward(self, images, captions=None):
        visual_features = self.encoder(images)
        
        # Training: Sử dụng captions làm đầu vào
        if captions is not None:
            outputs = self.decoder(visual_features, captions)
            return outputs
        # Inference: Tự sinh caption
        else:
            outputs = self.decoder(visual_features)
            return outputs