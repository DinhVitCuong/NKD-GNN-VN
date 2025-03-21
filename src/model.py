import torch
import torch.nn as nn
import torchvision.models as models

####################################################
# 1. ImageEncoder: Trích xuất đặc trưng ảnh
####################################################
class ImageEncoder(nn.Module):
    def __init__(self, embed_size=256, pretrained=True):
        super(ImageEncoder, self).__init__()
        # Tải ResNet-50
        self.cnn = models.resnet50(pretrained=pretrained)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, images):
        """
        images: Tensor (batch_size, 3, H, W)
        return: Tensor (batch_size, embed_size)
        """
        return self.cnn(images)

####################################################
# 2. CaptionDecoder: Dùng Embedding + LSTM
####################################################
class CaptionDecoder(nn.Module):
    def __init__(self, embed_size=256, hidden_size=512,
                 start_token=1, end_token=2,
                 vocab_size=5000, num_layers=1):
        super().__init__()
        
        # Embedding thường (không sigmoid)
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, vocab_size)

        self.start_token = start_token
        self.end_token   = end_token
        
    def forward(self, features, captions=None, max_len=20):
        """
        features: (batch_size, embed_size) từ ImageEncoder
        captions: (batch_size, seq_len) nếu training
        max_len:  độ dài tối đa khi inference
        """
        batch_size = features.size(0)

        # Khởi tạo hidden state (h) từ features
        # (num_layers=1, batch_size, hidden_size)
        h = features.unsqueeze(0)
        c = torch.zeros_like(h)

        # ============= TRAINING MODE =============
        if captions is not None:
            # 1) Lấy embedding của captions
            embeddings = self.embed(captions)  # (batch_size, seq_len, embed_size)
            # 2) LSTM
            lstm_out, _ = self.lstm(embeddings, (h, c))  # (batch_size, seq_len, hidden_size)
            # 3) Linear -> vocab_size
            outputs = self.fc(lstm_out)                  # (batch_size, seq_len, vocab_size)
            return outputs
        
        # ============= INFERENCE MODE =============
        else:
            input_words = torch.full((batch_size,), self.start_token, device=features.device)
            outputs = []

            for _ in range(max_len):
                # Lấy embedding của word hiện tại
                emb = self.embed(input_words).unsqueeze(1)  # (batch_size, 1, embed_size)
                
                # LSTM
                lstm_out, (h, c) = self.lstm(emb, (h, c))    # (batch_size, 1, hidden_size)
                preds = self.fc(lstm_out.squeeze(1))         # (batch_size, vocab_size)

                # Chọn từ có xác suất cao nhất
                next_words = preds.argmax(dim=1)
                outputs.append(next_words)
                input_words = next_words

                # Dừng sớm nếu gặp token end_token
                if (next_words == self.end_token).all():
                    break

            # Kết quả: (batch_size, seq_len_inference)
            return torch.stack(outputs, dim=1)

####################################################
# 3. Tổng hợp thành ImageCaptionModel
####################################################
class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size=256, hidden_size=512,
                 vocab_size=5000, start_token=1, end_token=2):
        super().__init__()
        self.encoder = ImageEncoder(embed_size=embed_size)
        self.decoder = CaptionDecoder(embed_size=embed_size,
                                      hidden_size=hidden_size,
                                      start_token=start_token,
                                      end_token=end_token,
                                      vocab_size=vocab_size)
        
    def forward(self, images, captions=None):
        """
        Nếu captions != None => Training mode
        Nếu captions == None => Inference mode
        """
        visual_features = self.encoder(images)
        if captions is not None:
            outputs = self.decoder(visual_features, captions=captions)
        else:
            outputs = self.decoder(visual_features, captions=None)
        return outputs
