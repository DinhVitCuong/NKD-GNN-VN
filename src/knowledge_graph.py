import networkx as nx
import stanza
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx

# Tải mô hình Stanza tiếng Việt
stanza.download('vi')
nlp = stanza.Pipeline('vi', processors='tokenize,ner')

# Hàm xây dựng đồ thị tri thức
def build_knowledge_graph(text, nlp_model):
    G = nx.Graph()
    entity_frequency = {}
    cooccurrence_frequency = {}
    word_segmented_sentences = []
    entity_dict = {}
    
    # Xử lý văn bản với Stanza
    doc = nlp_model(text)
    
    for sentence in doc.sentences:
        word_segmented_sentence = ' '.join([token.text for token in sentence.tokens])
        word_segmented_sentences.append(word_segmented_sentence)
        
        entities_in_sentence = []
        
        for entity in sentence.ents:
            entity_text = entity.text  # Không thay thế dấu cách bằng gạch dưới
            entity_type = entity.type
            entity_frequency[(entity_text, entity_type)] = entity_frequency.get((entity_text, entity_type), 0) + 1
            entities_in_sentence.append((entity_text, entity_type))
        
        # Ghi nhận sự xuất hiện đồng thời của các thực thể trong cùng câu
        for i, (entity1, type1) in enumerate(entities_in_sentence):
            for entity2, type2 in entities_in_sentence[i + 1:]:
                pair = ((entity1, type1), (entity2, type2))
                cooccurrence_frequency[pair] = cooccurrence_frequency.get(pair, 0) + 1
    
    # Cập nhật danh sách thực thể, giữ thực thể cuối cùng xuất hiện với frequency cập nhật
    for (entity_text, entity_type), frequency in entity_frequency.items():
        entity_dict[(entity_text, entity_type)] = (entity_text, entity_type, frequency)
    
    entity_list = list(entity_dict.values())
    
    # Xây dựng đồ thị từ dữ liệu thu thập được
    for entity, entity_type, frequency in entity_list:
        G.add_node(entity, frequency=frequency)
    
    for (entity1, type1), (entity2, type2) in cooccurrence_frequency.keys():
        cooccurrence_count = cooccurrence_frequency[((entity1, type1), (entity2, type2))]
        weight = cooccurrence_count / max(entity_frequency[(entity1, type1)], entity_frequency[(entity2, type2)])
        G.add_edge(entity1, entity2, weight=weight, cooccurrence_count=cooccurrence_count)
    
    return G, word_segmented_sentences, entity_list

# Hàm hiển thị đồ thị
def visualize_knowledge_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Hàm hiển thị thực thể và đồng xuất hiện
def print_entities_and_cooccurrences(G, entity_list):
    print("Entities:")
    for entity, entity_type, frequency in entity_list:
        print(f"Entity: {entity}, Type: {entity_type}, Frequency: {frequency}")
    
    print("\nCo-occurrences (Edges):")
    for edge in G.edges(data=True):
        print(f"Entity 1: {edge[0]}, Entity 2: {edge[1]}, Co-occurrence Count: {edge[2]['cooccurrence_count']}, Weight: {edge[2]['weight']}")

# Chuyển đổi đồ thị NetworkX sang Torch Geometric Data
def convert_networkx_to_data(G):
    data = from_networkx(G)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    return data

# # Ví dụ chạy thử
# if __name__ == "__main__":
#     text = "Sau hai thế kỷ, chính sách đóng cửa đất nước dưới thời Mạc phủ Tokugawa đã kết thúc khi Nhật Bản bị Hoa Kỳ ép mở cửa giao thương vào năm 1854. Những năm tiếp theo cuộc Minh Trị duy tân năm 1868 và sự sụp đổ của chế độ Mạc phủ, Nhật Bản đã chuyển đổi từ một xã hội phong kiến sang một quốc gia công nghiệp hiện đại."
    
#     G, word_segmented_sentences, entity_list = build_knowledge_graph(text, nlp)
    
#     print_entities_and_cooccurrences(G, entity_list)
#     visualize_knowledge_graph(G)