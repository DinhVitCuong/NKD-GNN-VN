import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import argparse
import os
from model import ImageCaptionModel, NKDGNN
from knowledge_graph import build_knowledge_graph
from template_caption import generate_template_caption, fill_template
from data_loader import CustomDataset, collate_fn
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
import stanza

# Khởi tạo PhoBERT tokenizer và model
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
stanza.download('vi')
stanza_model = stanza.Pipeline('vi', processors='tokenize,pos,ner')

def train(args):
    dataset = CustomDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    image_caption_model = ImageCaptionModel(embed_size=args.embed_size, hidden_size=args.hidden_size, vocab_size=tokenizer.vocab_size).to(args.device)
    nkdgnn = NKDGNN(input_dim=args.graph_dim, hidden_dim=args.hidden_size).to(args.device)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(list(image_caption_model.parameters()) + list(nkdgnn.parameters()), lr=args.lr)
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        for batch in dataloader:
            images = batch["images"].to(args.device)
            texts = batch["texts"]
            captions = batch["captions"]
            entity_lists = batch["entity_list"]
            graph_list = []
            entity_list = []
            entity_prob_list = []

            # 1. Sinh caption ban đầu
            generated_ids = image_caption_model(images)
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 2. Đục lỗ caption
            template_captions = [generate_template_caption(c, stanza_model) for c in generated_captions]
            
            # 3. Tạo đồ thị tri thưc
            for paragraph in texts:
                G, _, entities = build_knowledge_graph(paragraph, stanza_model)
                graph_data = from_networkx(G)
                graph_data.x = torch.eye(G.number_of_nodes(), dtype=torch.float) # One-hot encoding node features
                graph_list.append(graph_data)
                entity_list.append(entities) 
            # 4. Xử lý đồ thị tri thức bằng NKDGNN
            graph_data = [g.to(args.device) for g in graph_list]
            entity_list = [e.to(args.device) for e in entity_list]

            entity_prob_list, _ = nkdgnn(graph_data, entity_lists)
            
            # 5. Điền vào template
            filled_captions = []
            for template, entities in zip(template_captions, entity_prob_list):
                filled_captions.append(fill_template(template, entities))
            
            # 6. Tokenize captions
            outputs = tokenizer(filled_captions, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            output_ids = outputs["input_ids"].to(args.device)

            inputs = tokenizer(captions, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            caption_ids = inputs["input_ids"].to(args.device)
            
            # 7. Tính loss và update trọng số
            loss = criterion(output_ids.view(-1, tokenizer.vocab_size), caption_ids.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
        torch.save(image_caption_model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")
def evaluate(args):
    model = ImageCaptionModel(
        embed_size=args.embed_size,
        graph_dim=args.graph_dim,
        hidden_size=args.hidden_size,
        vocab_size=tokenizer.vocab_size
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
            decoded_captions = tokenizer.batch_decode(
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