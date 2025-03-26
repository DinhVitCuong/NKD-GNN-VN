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
from tqdm import tqdm 

from model import ImageCaptionModel
from GNN import NKDGNN
from knowledge_graph import build_knowledge_graph
from template_caption import generate_template_caption, fill_template
from data_loader import CustomDataset, collate_fn, load_vocab_index, encode, decode

# Khởi tạo PhoBERT tokenizer và model
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
# phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
stanza.download('vi')
stanza_model = stanza.Pipeline('vi', processors='tokenize,pos,ner')

def evaluate(models, dataloader, criterion, device):
    # Load vocab/index-to-word:
    vocab, embedding_to_word = load_vocab_index(args.vocab_path)
    image_caption_model, nkdgnn = models
    image_caption_model.eval()
    nkdgnn.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")  # Added tqdm
        for batch in progress_bar:
            images = batch["images"].to(device)
            texts = batch["texts"]
            captions = batch["captions"]
            entity_lists = batch["entity_list"]
            
            # 1. Generate initial captions
            generated_ids = image_caption_model(images)
            generated_ids = torch.argmax(generated_ids, dim=-1).squeeze(0).tolist()
            generated_captions = [decode(seq, embedding_to_word, skip_special_tokens=True) for seq in generated_ids]
            
            # 2. Mask captions
            template_captions = [generate_template_caption(c, stanza_model) for c in generated_captions]
            
            # 3. Build knowledge graphs
            graph_list = []
            entity_lists = []
            entity_prob_list = []
            for paragraph in texts:
                G, _, entities = build_knowledge_graph(paragraph, stanza_model)
                graph_list.append(G)
                entity_lists.append(entities) 
            # 4. Xử lý đồ thị tri thức bằng NKDGNN
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
            
            # 5. Fill templates
            filled_captions = []
            for template, entities in zip(template_captions, entity_prob_list):
                filled_captions.append(fill_template(template, entities))
            
            # 6. Compute loss
            output_ids = torch.tensor([encode(caption, vocab, max_length=64) for caption in filled_captions]).to(args.device)
            caption_ids = torch.tensor([encode(caption, vocab, max_length=64) for caption in captions]).to(args.device)
            
            loss = criterion(output_ids.view(-1), caption_ids.view(-1))
            total_loss += loss.item()
            progress_bar.set_postfix({'val_loss': loss.item()})  # Update progress bar
    
    return total_loss / len(dataloader)

def train(args):
    # Load vocab/index-to-word:
    vocab, embedding_to_word = load_vocab_index(args.vocab_path)
    # Load datasets
    train_dataset = CustomDataset(args.train_data_path)
    val_dataset = CustomDataset(args.val_data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Khởi tạo model
    image_caption_model = ImageCaptionModel(
        embed_size=args.embed_size, 
        hidden_size=args.hidden_size, 
        vocab_size=len(vocab)
    ).to(args.device)
    
    nkdgnn = NKDGNN(
        input_dim=1,
        hidden_dim=args.hidden_size,
        num_layers=2
    ).to(args.device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
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
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")  # Added tqdm
        for batch in progress_bar:
            images = batch["images"].to(args.device)
            texts = batch["texts"]
            captions = batch["captions"]
            graph_list = []
            entity_lists = []
            entity_prob_list = []

            # 1. Sinh caption ban đầu
            caption_ids = torch.tensor([encode(caption, vocab, max_length=64) for caption in captions]).to(args.device)
            generated_ids = image_caption_model(images, caption_ids)
            generated_ids = torch.argmax(generated_ids, dim=-1).squeeze(0).tolist()
            generated_captions = [decode(seq, embedding_to_word, skip_special_tokens=True) for seq in generated_ids]
            
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
                # print(f"GRAPH EDGES: \n {single_GA.edges}")
                # print(f"GRAPH NODES: \n {single_GA.nodes}")
                # print(f"ENTITY LIST: \n{single_EL}")
                node_mapping = {node: idx for idx, node in enumerate(single_GA.nodes())}
                mapped_edges = [[node_mapping[u], node_mapping[v]] for u, v in single_GA.edges()]
                edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous().to(args.device)
                # edge_index = torch.tensor(list(single_GA.edges)).t().contiguous().to(args.device)
                x = torch.tensor([single_GA.nodes[n]['frequency'] for n in single_GA.nodes]).float().unsqueeze(-1).to(args.device)
                node_degree = torch.tensor([single_GA.degree[n] for n in single_GA.nodes]).float().to(args.device)
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
            output_ids = torch.tensor([encode(caption, vocab, max_length=64) for caption in filled_captions]).to(args.device)
            caption_ids = torch.tensor([encode(caption, vocab, max_length=64) for caption in captions]).to(args.device)
            
            # 7. Tính loss và cập nhật trọng số
            loss = criterion(output_ids.view(-1), caption_ids.view(-1))
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
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "image_caption_model": image_caption_model.state_dict(),
                "nkdgnn": nkdgnn.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, args.checkpoint)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

def test(args):
    # Load vocab/index-to-word:
    vocab, embedding_to_word = load_vocab_index(args.vocab_path)
    # Load test dataset
    test_dataset = CustomDataset(args.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Khởi tạo model
    image_caption_model = ImageCaptionModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab)
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
        progress_bar = tqdm(test_loader, desc="Testing")  # Added tqdm
        for batch in progress_bar:
            images = batch["images"].to(args.device)
            texts = batch["texts"]
            captions = batch["captions"]
            entity_lists = batch["entity_list"]
            graph_list = []
            entity_list = []
            entity_prob_list = []

            # 1. Sinh caption ban đầu
            generated_ids = image_caption_model(images)
            generated_ids = torch.argmax(generated_ids, dim=-1).squeeze(0).tolist()
            generated_captions = [decode(seq, embedding_to_word, skip_special_tokens=True) for seq in generated_ids]
            
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
            
            # # 6. Tokenize captions
            # output_ids = torch.tensor([encode(caption, vocab, max_length=64) for caption in filled_captions]).to(args.device)
            # caption_ids = torch.tensor([encode(caption, vocab, max_length=64) for caption in captions]).to(args.device)
            
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
    
    # General parameters
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Data paths
    parser.add_argument("--train_data_path", help="Path to training JSON", required=True)
    parser.add_argument("--val_data_path", help="Path to validation JSON", required=True)
    parser.add_argument("--test_data_path", help="Path to test JSON", required=True)
    parser.add_argument("--output_file", default="results.json", required=True)
    
    # Model parameters
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")  # Added argument
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
        if args.test_data_path:
            print("Training completed. Running test...")
            test(args)
    else:
        test(args)
# Example
# # Train the model with early stopping and automatic testing
# python main.py --mode train \
#   --train_data_path train.json \
#   --val_data_path val.json \
#   --test_data_path test.json \
#   --vocab_path NKD-GNN-VN-main\src\vocab_split.json
#   --patience 5 \
#   --checkpoint best_model.pt

# # Test the model separately
# python main.py --mode test \
#   --vocab_path NKD-GNN-VN-main\src\vocab_split.json
#   --test_data_path test.json \
#   --checkpoint best_model.pt