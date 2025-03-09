import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import argparse
import os
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
import stanza


from model import ImageCaptionModel
from GNN import NKDGNN
from knowledge_graph import build_knowledge_graph
from template_caption import generate_template_caption, fill_template
from data_loader import CustomDataset, collate_fn

# Khởi tạo PhoBERT tokenizer và model
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
stanza.download('vi')
stanza_model = stanza.Pipeline('vi', processors='tokenize,pos,ner')

def evaluate(models, dataloader, criterion, device):
    image_caption_model, nkdgnn = models
    image_caption_model.eval()
    nkdgnn.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            captions = batch["captions"]
            entity_lists = batch["entity_list"]
            
            # 1. Sinh caption ban đầu
            generated_ids = image_caption_model(images)
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 2. Đục lỗ caption
            template_captions = [generate_template_caption(c, stanza_model) for c in generated_captions]
            
            # 3. Tạo đồ thị tri thức
            graph_list = []
            for paragraph in texts:
                G, _, entities = build_knowledge_graph(paragraph, stanza_model)
                graph_data = from_networkx(G)
                graph_data.x = torch.eye(G.number_of_nodes(), dtype=torch.float)
                graph_list.append(graph_data)
            
            # 4. Xử lý đồ thị tri thức
            graph_data = [g.to(device) for g in graph_list]
            entity_lists = [e.to(device) for e in entity_lists]
            
            entity_prob_list, _ = nkdgnn(graph_data, entity_lists)
            
            # 5. Điền vào template
            filled_captions = []
            for template, entities in zip(template_captions, entity_prob_list):
                filled_captions.append(fill_template(template, entities))
            
            # 6. Tính loss
            outputs = tokenizer(filled_captions, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            output_ids = outputs["input_ids"].to(device)
            
            inputs = tokenizer(captions, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            caption_ids = inputs["input_ids"].to(device)
            
            loss = criterion(output_ids.view(-1, tokenizer.vocab_size), caption_ids.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(args):
    # Load datasets
    train_dataset = CustomDataset(args.train_data_path)
    val_dataset = CustomDataset(args.val_data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Khởi tạo model
    image_caption_model = ImageCaptionModel(
        embed_size=args.embed_size, 
        hidden_size=args.hidden_size, 
        vocab_size=tokenizer.vocab_size
    ).to(args.device)
    
    nkdgnn = NKDGNN(
        input_dim=1,
        hidden_dim=args.hidden_size,
        num_layers=2
    ).to(args.device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(
        list(image_caption_model.parameters()) + list(nkdgnn.parameters()), 
        lr=args.lr
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        image_caption_model.train()
        nkdgnn.train()
        train_loss = 0
        
        
        for batch in train_loader:
            images = batch["images"].to(args.device)
            texts = batch["texts"]
            captions = batch["captions"]
            graph_list = []
            entity_lists = []
            entity_prob_list = []

            # 1. Sinh caption ban đầu
            generated_ids = image_caption_model(images)
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 2. Đục lỗ caption
            template_captions = [generate_template_caption(c, stanza_model) for c in generated_captions]
            
            # 3. Tạo đồ thị tri thưc
            for paragraph in texts:
                G, _, entities = build_knowledge_graph(paragraph, stanza_model)
                graph_list.append(G)
                entity_lists.append(entities) 
            # 4. Xử lý đồ thị tri thức bằng NKDGNN
            # graph_data = [g.to(args.device) for g in graph_list]
            # entity_lists = [e.to(args.device) for e in entity_lists]
            for single_GA, single_EL in zip(graph_list, entity_lists):
                edge_index = torch.tensor(list(single_GA.edges)).t().contiguous().to(args.device)
                x = torch.tensor([single_GA.nodes[n]['frequency'] for n in single_GA.nodes]).float().unsqueeze(-1).to(args.device)
                node_degree = torch.tensor([G.degree[n] for n in single_GA.nodes]).float().to(args.device)
                ent_prob = nkdgnn(x, edge_index, node_degree)
                results = []
                for node, prob in zip(single_GA.nodes(), ent_prob):
                    entity_info = next((e for e in single_EL if e[0] == node), None)
                    if entity_info:
                        results.append((entity_info[0], entity_info[1], prob.item()))
                
                entity_prob_list.append(sorted(results, key=lambda x: -x[2]))
            
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
        
        # Validation
        val_loss = evaluate(
            (image_caption_model, nkdgnn), 
            val_loader, 
            criterion, 
            args.device
        )
        
        print(f"Epoch {epoch+1}")
        print(f"Train loss: {train_loss/len(train_loader):.4f}")
        print(f"Val loss: {val_loss:.4f}")
        
        # Lưu model tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "image_caption_model": image_caption_model.state_dict(),
                "nkdgnn": nkdgnn.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, args.checkpoint)

def test(args):
    # Load test dataset
    test_dataset = CustomDataset(args.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Khởi tạo model
    image_caption_model = ImageCaptionModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=tokenizer.vocab_size
    ).to(args.device)
    
    nkdgnn = NKDGNN(
        input_dim=args.graph_dim,
        hidden_dim=args.hidden_size
    ).to(args.device)
    
    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    image_caption_model.load_state_dict(checkpoint["image_caption_model"])
    nkdgnn.load_state_dict(checkpoint["nkdgnn"])
    
    image_caption_model.eval()
    nkdgnn.eval()
    
    results = []
    
    with torch.no_grad():
        
        for batch in test_loader:
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
            
            # # 7. Tính loss và update trọng số
            # loss = criterion(output_ids.view(-1, tokenizer.vocab_size), caption_ids.view(-1))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # Lưu kết quả
            for img_id, caption in zip(batch["image_ids"], filled_captions):
                results.append({
                    "image_id": img_id,
                    "caption": caption
                })
    
    # Lưu file JSON
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Tham số chung
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Đường dẫn dữ liệu
    parser.add_argument("--train_data_path", help="Path to training JSON")
    parser.add_argument("--val_data_path", help="Path to validation JSON")
    parser.add_argument("--test_data_path", help="Path to test JSON")
    parser.add_argument("--output_file", default="results.json")
    
    # Tham số model
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    
    # Tham số huấn luyện
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--checkpoint", default="best_model.pt")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    else:
        test(args)