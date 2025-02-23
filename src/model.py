import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

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

class NKDGNN(nn.Module):
    """
    Neural Knowledge-Driven GNN với cải tiến attention
    """
    def __init__(self, visual_dim, graph_dim, hidden_dim, num_heads=4):
        super().__init__()
        
        # Graph Processing
        self.gat1 = GATConv(graph_dim, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim*num_heads, hidden_dim, heads=1)
        
        # Visual Projection
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Cross-modal Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # Fusion Layer
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, visual_feats, graph_data):
        # Xử lý đồ thị
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        graph_feats = global_mean_pool(x, batch)  # (batch_size, hidden_dim)
        
        # Xử lý ảnh
        visual_feats = F.relu(self.visual_proj(visual_feats))  # (batch_size, hidden_dim)
        
        # Cross-modal Attention
        attn_output, _ = self.attention(
            visual_feats.unsqueeze(0),
            graph_feats.unsqueeze(0),
            graph_feats.unsqueeze(0)
        )
        
        # Hợp nhất đặc trưng
        combined = torch.cat([visual_feats, attn_output.squeeze(0)], dim=1)
        fused_features = F.relu(self.fusion(combined))
        
        return fused_features

class CaptionDecoder(nn.Module):
    """
    Module tạo caption sử dụng LSTM với cơ chế attention
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, teacher_forcing_ratio=0.5):
        batch_size = features.size(0)
        embeddings = self.embed(captions)  # (batch_size, seq_len, embed_size)
        hiddens = []
        h = features.unsqueeze(0)  # (num_layers, batch_size, hidden_size)
        c = torch.zeros_like(h)
        
        # Tạo caption từng từ
        for t in range(captions.size(1)-1):
            # Tính attention weights
            attention_weights = F.softmax(self.attention(h[-1]), dim=1)  # (batch_size, 1)
            
            # Context vector
            context = attention_weights * features  # (batch_size, hidden_size)
            
            # Ghép embedding và context
            lstm_input = torch.cat([embeddings[:,t], context], dim=1).unsqueeze(1)
            
            # LSTM step
            out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Dự đoán từ
            outputs = self.fc(out.squeeze(1))
            hiddens.append(outputs)
            
            # Teacher forcing
            if torch.rand(1) < teacher_forcing_ratio:
                next_word = captions[:, t+1]
            else:
                next_word = outputs.argmax(1)
                
            embeddings[:, t+1] = self.embed(next_word)
            
        return torch.stack(hiddens, dim=1)

class ImageCaptionModel(nn.Module):
    """
    Mô hình end-to-end hoàn chỉnh
    """
    def __init__(self, embed_size, graph_dim, hidden_size, vocab_size):
        super().__init__()
        
        # Các thành phần
        self.encoder = ImageEncoder(embed_size)
        self.nkdgnn = NKDGNN(embed_size, graph_dim, hidden_size)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size)
        
    def forward(self, images, graph_data, captions=None, teacher_forcing_ratio=0.5):
        # Trích xuất đặc trưng
        visual_features = self.encoder(images)  # (batch_size, embed_size)
        
        # Xử lý đồ thị tri thức
        fused_features = self.nkdgnn(visual_features, graph_data)  # (batch_size, hidden_size)
        
        # Tạo caption
        if captions is not None:
            outputs = self.decoder(fused_features, captions, teacher_forcing_ratio)
        else:
            # Inference mode
            outputs = self.generate(fused_features)
            
        return outputs
    
    def generate(self, features, max_len=20):
        # Tạo caption từ đầu
        batch_size = features.size(0)
        captions = [torch.zeros(batch_size, dtype=torch.long).to(features.device)]
        
        h = features.unsqueeze(0)
        c = torch.zeros_like(h)
        
        for _ in range(max_len):
            embeddings = self.decoder.embed(captions[-1]).unsqueeze(1)
            context = features.unsqueeze(1)
            lstm_input = torch.cat([embeddings, context], dim=2)
            
            out, (h, c) = self.decoder.lstm(lstm_input, (h, c))
            outputs = self.decoder.fc(out.squeeze(1))
            captions.append(outputs.argmax(dim=1))
            
        return torch.stack(captions[1:], dim=1)

