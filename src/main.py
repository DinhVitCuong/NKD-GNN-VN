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

def train(args):
    # Khởi tạo model
    model = ImageCaptionModel(
        embed_size=args.embed_size,
        graph_dim=args.graph_dim,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size
    ).to(args.device)
    
    # Chuẩn bị dữ liệu
    dataset = CustomDataset(args.data_path, nlp)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Huấn luyện
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = batch["images"].to(args.device)
            graph_data = batch["graph_data"].to(args.device)
            captions = batch["captions"]
            
            # Tokenize captions bằng stanza
            tokenized_captions = []
            for caption in captions:
                doc = nlp(caption)
                tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
                tokenized_captions.append(" ".join(tokens))
            
            # Chuyển captions thành tensor (giả sử đã có vocab)
            caption_ids = preprocess_captions(tokenized_captions).to(args.device)
            
            # Forward pass
            outputs = model(images, graph_data, caption_ids)
            loss = criterion(outputs.view(-1, args.vocab_size), caption_ids.view(-1))
            
            # Backward và tối ưu hóa
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
        vocab_size=args.vocab_size
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    dataset = CustomDataset(args.data_path, nlp)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    results = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(args.device)
            graph_data = batch["graph_data"].to(args.device)
            
            # Sinh caption
            generated = model.generate(images, graph_data)
            
            # Giải mã tokens thành văn bản
            decoded_captions = [decode(caption) for caption in generated]
            results.extend(decoded_captions)
    
    # Lưu kết quả
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def preprocess_captions(captions, max_length=20):
    """
    Chuyển đổi captions thành tensor sử dụng stanza tokenizer
    """
    tokenized_captions = []
    for caption in captions:
        doc = nlp(caption)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        tokenized_captions.append(" ".join(tokens))
    
    # Giả sử đã có vocab và hàm chuyển đổi từ tokens sang ids
    # Ví dụ: caption_ids = [vocab[token] for token in tokens]
    # Trong ví dụ này, tôi giả sử bạn đã có sẵn vocab và hàm chuyển đổi
    caption_ids = torch.tensor([vocab[token] for token in tokenized_captions], dtype=torch.long)
    return caption_ids

def decode(caption_ids):
    """
    Giải mã caption_ids thành văn bản
    """
    # Giả sử đã có vocab và hàm chuyển đổi ngược từ ids sang tokens
    tokens = [vocab.idx2word[id.item()] for id in caption_ids]
    return " ".join(tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Tham số chung
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Đường dẫn dữ liệu
    parser.add_argument("--data_path", required=True, help="Đường dẫn đến file JSON dữ liệu")
    parser.add_argument("--output_file", default="results.json")
    
    # Tham số model
    parser.add_argument("--embed_size", type=int, default=2048)
    parser.add_argument("--graph_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10000)
    
    # Tham số huấn luyện
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", default="best_model.pt")
    
    args = parser.parse_args()
    
    # Khởi tạo NLP pipeline (stanza)
    stanza.download("vi")
    nlp = stanza.Pipeline("vi", processors="tokenize")
    
    # Khởi tạo vocab (giả sử đã có)
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, ...}  # Thêm từ vựng của bạn vào đây
    vocab.idx2word = {idx: word for word, idx in vocab.items()}
    
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)