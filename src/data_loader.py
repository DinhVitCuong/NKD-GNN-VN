import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import argparse
import os
from torch_geometric.data import Batch
Image.MAX_IMAGE_PIXELS = None

class CustomDataset(Dataset):
    def __init__(self, data_path, max_paragraphs=3):
        with open(data_path) as f:
            self.data = json.load(f)
        self.items = list(self.data.values())
        self.max_paragraphs = max_paragraphs
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Đọc ảnh từ đường dẫn trong JSON
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Chọn các đoạn văn có điểm cao nhất
        sorted_paragraphs = sorted(
            zip(item["paragraphs"], item["scores"]),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_paragraphs]
        
        # Kết hợp các đoạn văn thành một văn bản
        combined_text = " ".join([p[0] for p in sorted_paragraphs])
        
        #Đọc caption gốc:
        caption = item["caption"]
        # # Xây dựng đồ thị tri thức
        # G, _, entity_list = build_knowledge_graph(combined_text, self.nlp_model)
        
        # # Chuyển đổi đồ thị sang Torch Geometric Data
        # graph_data = from_networkx(G)
        # graph_data.x = torch.eye(G.number_of_nodes(), dtype=torch.float)  # One-hot encoding node features
        
        return {
            "image": image,
            "caption": caption,
            "text": combined_text
        }

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    text = [item["text"] for item in batch]
    return {
        "images": images,
        "texts": text,
        "captions": captions
    }
    
def load_vocab_embedding(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab)} tokens from {filepath}")
    # Create a mapping from embedding (converted to a tuple) to the word.
    embedding_to_word = {tuple(embedding): word for word, embedding in vocab.items()}
    return vocab, embedding_to_word
    
def load_vocab_index(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab)} tokens from {filepath}")
    # Create a mapping from embedding (converted to a tuple) to the word.
    index_to_word = {index: word for word, index in vocab.items()}
    return vocab, index_to_word

# Hàm encode: chuyển chuỗi thành danh sách token id
def encode(text, vocab, max_length=64, pad_token="<pad>", start_token="<start>", end_token="<end>"):
    tokens = text.split()  # Dùng split đơn giản (bạn có thể thay bằng tokenizer chuyên biệt nếu cần)
    tokens = [start_token] + tokens + [end_token]
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    if len(token_ids) < max_length:
        token_ids += [vocab[pad_token]] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]
    return token_ids

# Hàm decode: chuyển danh sách token id thành chuỗi văn bản
def decode(token_ids, index_to_word, skip_special_tokens=True, special_tokens=["<unk>", "<start>", "<end>", "<pad>"]):
    tokens = []
    for id in token_ids:
        token = index_to_word.get(int(id), "<unk>")
        if skip_special_tokens and token in special_tokens:
            continue
        tokens.append(token)
    return " ".join(tokens)