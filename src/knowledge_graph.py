from vncorenlp import VnCoreNLP
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize VnCoreNLP: Set the correct path to the VnCoreNLP models jar file
VNLPCORENLP_PATH = r'E:\DATN\NKD-GNN-test\VnCoreNLP\VnCoreNLP-1.2.jar'  # Replace with your correct path
rdrsegmenter = VnCoreNLP(VNLPCORENLP_PATH, annotators="wseg,pos,ner", max_heap_size='-Xmx2g')

# Function to extract named entities from text
def extract_entities(text):
    annotations = rdrsegmenter.annotate(text)
    entities = defaultdict(list)  # Dictionary to store entities with their types

    for sentence in annotations['sentences']:
        for token in sentence:
            ner_tag = token['nerLabel']
            word = token['form']
            if ner_tag.startswith('B-') or ner_tag.startswith('I-'):
                entity_type = ner_tag.split('-')[1]
                entities[entity_type].append(word)

    return entities

# Function to build the knowledge graph based on co-occurrence of entities
def build_knowledge_graph(text):
    annotations = rdrsegmenter.annotate(text)
    G = nx.Graph()

    # Track occurrences and co-occurrences
    entity_occurrences = defaultdict(int)  # To count occurrences of each entity
    co_occurrences = defaultdict(int)      # To count co-occurrences of entity pairs

    # Iterate over each sentence and find entities
    for sentence in annotations['sentences']:
        sentence_entities = set()  # Store unique entities in a sentence
        
        # Collect entities in this sentence
        for token in sentence:
            ner_tag = token['nerLabel']
            word = token['form']
            if ner_tag.startswith('B-') or ner_tag.startswith('I-'):
                entity_type = ner_tag.split('-')[1]
                sentence_entities.add(word)
                entity_occurrences[word] += 1  # Count occurrences of each entity

        # Create co-occurrences between all entities in the same sentence
        sentence_entities = list(sentence_entities)
        for i in range(len(sentence_entities)):
            for j in range(i + 1, len(sentence_entities)):
                entity1 = sentence_entities[i]
                entity2 = sentence_entities[j]
                
                if entity1 != entity2:
                    co_occurrences[(entity1, entity2)] += 1
                    co_occurrences[(entity2, entity1)] += 1  # Symmetric

    # Build the graph with weighted edges
    for (entity1, entity2), co_count in co_occurrences.items():
        max_occurrence = max(entity_occurrences[entity1], entity_occurrences[entity2])
        weight = co_count / max_occurrence  # Calculate the weight based on the formula

        # Add the edge to the graph
        if not G.has_edge(entity1, entity2):
            G.add_edge(entity1, entity2, weight=weight)

    return G, entity_occurrences, co_occurrences

# Function to visualize the knowledge graph
def visualize_knowledge_graph(G):
    pos = nx.spring_layout(G)  # Positioning the nodes
    plt.figure(figsize=(12, 12))
    
    # Drawing the graph with labels
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
    
    # Drawing edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()

# Example usage
if __name__ == "__main__":
    vietnamese_text = """Nguyễn Phú Trọng đã phát biểu tại tòa nhà Quốc hội và trường Đại học Quốc gia Hà Nội.
                         Hội nghị được tổ chức bởi Chính phủ và Đảng Cộng sản Việt Nam."""

    # Extract entities
    entities = extract_entities(vietnamese_text)
    print("Extracted Entities:", entities)

    # Build the knowledge graph
    G, entity_occurrences, co_occurrences = build_knowledge_graph(vietnamese_text)

    print("Entity Occurrences:", entity_occurrences)
    print("Co-Occurrences:", co_occurrences)

    # Visualize the knowledge graph
    visualize_knowledge_graph(G)
