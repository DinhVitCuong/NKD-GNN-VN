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
from data_loader import CustomDataset, collate_fn
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch_geometric.data import Batch

# Khởi tạo PhoBERT tokenizer và model
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")

def train(args):
    # Khởi tạo model
    model = ImageCaptionModel(
        embed_size=args.embed_size,
        graph_dim=args.graph_dim,
        hidden_size=args.hidden_size,
        vocab_size=phobert_tokenizer.vocab_size  # Sử dụng vocab size từ PhoBERT
    ).to(args.device)
    
    # Chuẩn bị dữ liệu
    dataset = CustomDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Loss và optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=phobert_tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Huấn luyện
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = batch["images"].to(args.device)
            graph_data = batch["graph_data"].to(args.device)
            captions = batch["captions"]
            
            # Tokenize bằng PhoBERT
            inputs = phobert_tokenizer(
                captions,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt"
            )
            caption_ids = inputs["input_ids"].to(args.device)
            
            # Forward pass
            outputs = model(images, graph_data, caption_ids)
            loss = criterion(outputs.view(-1, phobert_tokenizer.vocab_size), caption_ids.view(-1))
            
            # Backward và optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")

def evaluate(args):
    model = ImageCaptionModel(
        embed_size=args.embed_size,
        graph_dim=args.graph_dim,
        hidden_size=args.hidden_size,
        vocab_size=phobert_tokenizer.vocab_size
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    dataset = CustomDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    results = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(args.device)
            graph_data = batch["graph_data"].to(args.device)
            
            # Sinh caption
            generated_ids = model.generate(images, graph_data)
            
            # Giải mã tokens
            decoded_captions = phobert_tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            results.extend(decoded_captions)
    
    # Lưu kết quả
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Tham số chung
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Đường dẫn dữ liệu
    parser.add_argument("--data_path", required=True, help="Path to JSON data file")
    parser.add_argument("--output_file", default="results.json")
    
    # Tham số model
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--graph_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    
    # Tham số huấn luyện
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)  # Learning rate nhỏ hơn cho fine-tuning
    parser.add_argument("--checkpoint", default="best_model.pt")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)