from vncorenlp import VnCoreNLP
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
py_vncorenlp.download_model(save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

# Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models`
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "ner"], save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

# Define placeholder tags
PERSON_PLACEHOLDER = "<Person>"
PLACE_PLACEHOLDER = "<Place>"
ORGANIZATION_PLACEHOLDER = "<Organization>"
BUILDING_PLACEHOLDER = "<Building>"

# Define a list of common building-related words in Vietnamese
building_keywords = [
    "nhà", "tòa nhà", "cao ốc", "trụ sở", "công trình", "trung tâm", "khu chung cư", "biệt thự",
    "nhà máy", "nhà xưởng", "nhà kho", "bệnh viện", "trường học", "đại học", "khách sạn", "siêu thị",
    "chung cư", "nhà thờ", "chùa", "thánh đường", "nhà ga", "sân bay", "trung tâm thương mại", "nhà hát",
    "bảo tàng", "khu hành chính", "nhà hàng", "văn phòng", "khu nghỉ dưỡng", "ký túc xá", "bưu điện",
    "phòng khám", "thư viện", "sân vận động", "tòa thị chính", "trạm cứu hỏa", "cục cảnh sát", "trạm xăng",
    "công viên", "nhà nghỉ", "nhà thi đấu", "khu công nghiệp"
]

# Function to detect and replace building-related words
def get_building_placeholder(word):
    if word.lower() in building_keywords:
        return BUILDING_PLACEHOLDER
    return word

# Function to extract named entities from text
def build_knowledge_graph(text):
    """
    This function takes a text input, extracts entities (including <BUILDING> types),
    and builds a knowledge graph based on entity co-occurrence.
    """
    
    # Step 1: Entity Extraction (Named Entity Recognition + Building Detection)
    annotations = model.annotate_text(text)
    entities = []
    sentence_entities = []  # Store entities found in each sentence for co-occurrence
    for sentence_key in annotations:
        sentence = annotations[sentence_key]
        sentence_entity_set = set()
        # Process each token in the sentence
        for token in sentence:
            word = token['wordForm']
            ner_tag = token['nerLabel']

            # If word is a recognized named entity by VnCoreNLP
            if ner_tag.startswith('B-') or ner_tag.startswith('I-'):
                entities.append((word, ner_tag.split('-')[1]))  # Add entity and its type
                sentence_entity_set.add(word)  # Add to sentence entity set
            
            # If word is in building keywords, tag it as <BUILDING>
            elif word.lower() in building_keywords:
                entities.append((word, '<BUILDING>'))
                sentence_entity_set.add(word)  # Add building to sentence entity set

        # Add the sentence entity set to track co-occurrence
        sentence_entities.append(sentence_entity_set)

    # Step 2: Co-occurrence Detection
    co_occurrences = []
    for entity_set in sentence_entities:
        entity_list = list(entity_set)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                co_occurrences.append((entity_list[i], entity_list[j]))

    # Step 3: Building the Knowledge Graph
    G = nx.Graph()

    # Add nodes (entities)
    for entity in entities:
        G.add_node(entity[0], label=entity[1])  # Add entity name and its type (label)
    
    # Add edges (relationships based on co-occurrence)
    for entity1, entity2 in co_occurrences:
        if entity1 != entity2:
            if G.has_edge(entity1, entity2):
                G[entity1][entity2]['weight'] += 1  # Increment weight for repeated co-occurrences
            else:
                G.add_edge(entity1, entity2, weight=1)  # Add edge with initial weight

    return G

# Function to visualize the knowledge graph
def visualize_knowledge_graph(G):
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example text input
    text = """Nguyễn Phú Trọng phát biểu tại tòa nhà Quốc hội và gặp gỡ các đại biểu đến từ nhiều bệnh viện tại Hà Nội."""
    
    # Build the knowledge graph from text
    G = build_knowledge_graph(text)
    
    # Visualize the resulting knowledge graph
    visualize_knowledge_graph(G)