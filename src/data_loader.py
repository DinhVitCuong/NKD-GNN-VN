import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import argparse
import os
from model import ImageCaptionModel
from knowledge_graph import build_knowledge_graph
from template_caption import generate_template_caption
import stanza
from torch_geometric.data import Batch

class CustomDataset(Dataset):
    def __init__(self, data_path, nlp_model, max_paragraphs=3):
        with open(data_path) as f:
            self.data = json.load(f)
        self.items = list(self.data.values())
        self.nlp_model = nlp_model
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
        
        # Xây dựng đồ thị tri thức
        G, _, _ = build_knowledge_graph(combined_text, self.nlp_model)
        
        # Thêm batch index cho đồ thị
        G.batch = torch.zeros(G.num_nodes, dtype=torch.long)
        
        # Tạo template từ caption chính
        template = generate_template_caption(item["caption"], self.nlp_model)
        
        return {
            "image": image,
            "graph_data": G,
            "caption": item["caption"],
            "template": template
        }

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    templates = [item["template"] for item in batch]
    graph_data = [item["graph_data"] for item in batch]
    
    # Chuyển đổi đồ thị sang định dạng Batch của PyG
    graph_batch = Batch.from_data_list(graph_data)
    
    return {
        "images": images,
        "graph_data": graph_batch,
        "captions": captions,
        "templates": templates
    }

