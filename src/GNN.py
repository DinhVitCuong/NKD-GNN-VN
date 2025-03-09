import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

class NKDGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Gate parameters (GRU-like)
        self.W_z = nn.ModuleList([nn.Linear(input_dim + hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.W_r = nn.ModuleList([nn.Linear(input_dim + hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.W_h = nn.ModuleList([nn.Linear(input_dim + hidden_dim, hidden_dim) for _ in range(num_layers)])
        
        # Attention parameters
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        
        # Final transformation
        self.W3 = nn.Linear(2*hidden_dim, hidden_dim)

    def forward(self, x, edge_index, node_degree):
        # Initialize hidden states
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        # Message passing with GRU-like gates
        for layer in range(self.num_layers):
            # Aggregate neighbor messages (Eq.2)
            row, col = edge_index
            msg = scatter(h[col], row, dim=0, reduce='mean')  # Mean aggregation
            
            # Gate computations (Eq.3-6)
            combined = torch.cat([x, msg], dim=-1)
            
            z = torch.sigmoid(self.W_z[layer](combined))  # Update gate (Eq.3)
            r = torch.sigmoid(self.W_r[layer](combined))  # Reset gate (Eq.4)
            
            h_candidate = torch.tanh(self.W_h[layer](
                torch.cat([x, r * msg], dim=-1)
            ))  # Candidate state (Eq.5)
            
            h = (1 - z) * h + z * h_candidate  # Final update (Eq.6)
        
        # Attention mechanism (Eq.7-8)
        key_idx = torch.argmax(node_degree)
        N_b = h[key_idx]
        
        scores = self.q(torch.tanh(self.W1(N_b.unsqueeze(0)) + self.W2(h)))
        alpha = F.softmax(scores, dim=0)
        N_g = torch.sum(alpha * h, dim=0)
        
        # Final representation (Eq.9)
        N_r = self.W3(torch.cat([N_g, N_b]))
        
        # Entity probabilities (Eq.10-11)
        logits = torch.mm(h, N_r.unsqueeze(-1)).squeeze()
        probs = F.softmax(logits, dim=0)
        
        return probs

# def run_nkd_gnn(G, entity_list):
#     # Convert graph to tensors
#     edge_index = torch.tensor(list(G.edges)).t().contiguous()
#     x = torch.tensor([G.nodes[n]['frequency'] for n in G.nodes]).float().unsqueeze(-1)
#     node_degree = torch.tensor([G.degree[n] for n in G.nodes]).float()
    
#     # Initialize model
#     model = NKDGNN(input_dim=1, hidden_dim=64, num_layers=2)
    
#     # Get probabilities
#     probs = model(x, edge_index, node_degree)
    
#     # Map to entities
#     results = []
#     for node, prob in zip(G.nodes(), probs):
#         entity_info = next((e for e in entity_list if e[0] == node), None)
#         if entity_info:
#             results.append((entity_info[0], entity_info[1], prob.item()))
    
#     return sorted(results, key=lambda x: -x[2])

# class NKDGNN(nn.Module):
#     """
#     News Knowledge-Driven Graph Neural Network (NKD-GNN)
#     Dùng để phân tích quan hệ giữa các thực thể trong đồ thị tri thức và tính xác suất của các thực thể
#     """
#     def __init__(self, input_dim, hidden_dim, num_heads=4):
#         super().__init__()
        
#         # GNN Layers
#         self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
#         self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        
#         # Attention Mechanism
#         self.attention = nn.Linear(hidden_dim, 1)
        
#         # Final Prediction Layer
#         self.fc = nn.Linear(hidden_dim, 1)

#     def forward(self, data, entity_list):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
        
#         # GAT Layers
#         x = F.relu(self.gat1(x, edge_index))
#         x = F.relu(self.gat2(x, edge_index))
        
#         # Compute Global Graph Representation
#         attention_weights = F.softmax(self.attention(x), dim=0)
#         global_representation = torch.sum(attention_weights * x, dim=0)
        
#         # Compute Entity Probabilities
#         entity_scores = self.fc(x)
#         entity_probs = torch.sigmoid(entity_scores).squeeze(-1)
        
#         # Tạo danh sách thực thể với xác suất
#         entity_prob_list = [(entity_list[i][0], entity_list[i][1], entity_probs[i].item()) for i in range(len(entity_list))]
        
#         return entity_prob_list, global_representation