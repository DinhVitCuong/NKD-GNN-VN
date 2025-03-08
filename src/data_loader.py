import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import argparse
import os
from torch_geometric.data import Batch

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
