import networkx as nx
import py_vncorenlp
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx

# Automatically download VnCoreNLP components from the original repository
py_vncorenlp.download_model(save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

# Load VnCoreNLP from the specified folder
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "ner"], save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP',max_heap_size='-Xmx4g')

# Function to extract named entities and build the knowledge graph
def build_knowledge_graph(text):
    # Create an empty graph
    G = nx.Graph()

    # Dictionary to track entity frequency (how many times an entity appears)
    entity_frequency = {}

    # Dictionary to track co-occurrence frequency between entity pairs
    cooccurrence_frequency = {}

    # Initialize list to store the segmented sentences
    word_segmented_sentences = []
    # Split into sentences
    sentences = text.split(".")

    # Annotate the chunk with VnCoreNLP
    for splited_sentence in sentences:
        annotations = model.annotate_text(splited_sentence)

        for sentence_key in annotations:
            sentence = annotations[sentence_key]

            # Store the word-segmented sentence
            word_segmented_sentence = ' '.join([token['wordForm'] for token in sentence])
            word_segmented_sentences.append(word_segmented_sentence)

            # Initialize a list for the current sentence's entities
            entities_in_sentence = []
            current_entity = None
            current_type = None

            # Loop through each token in the sentence to identify entities
            for token in sentence:
                word = token['wordForm']
                ner_tag = token['nerLabel']  # Get NER label from the token

                # If it's the beginning of a new entity (B-*)
                if ner_tag.startswith('B-'):
                    if current_entity:
                        # Add the previous entity
                        entities_in_sentence.append((current_entity.strip(), current_type))
                        # Track frequency of the entity
                        entity_frequency[current_entity.strip()] = entity_frequency.get(current_entity.strip(), 0) + 1

                    # Start a new entity
                    current_entity = word
                    current_type = ner_tag[2:]
                # If it's the continuation of an entity (I-*)
                elif ner_tag.startswith('I-') and current_entity:
                    current_entity += f"_{word}"
                # If it's not an entity, finalize the current entity
                else:
                    if current_entity:
                        entities_in_sentence.append((current_entity.strip(), current_type))
                        # Track frequency of the entity
                        entity_frequency[current_entity.strip()] = entity_frequency.get(current_entity.strip(), 0) + 1
                        current_entity = None
                        current_type = None

            # Add the last entity in the sentence if it exists
            if current_entity:
                entities_in_sentence.append((current_entity.strip(), current_type))
                # Track frequency of the entity
                entity_frequency[current_entity.strip()] = entity_frequency.get(current_entity.strip(), 0) + 1

            # Now track co-occurrence for entities in this sentence
            for i, (entity1, type1) in enumerate(entities_in_sentence):
                for entity2, type2 in entities_in_sentence[i + 1:]:
                    # Track co-occurrence frequency between entity pairs
                    pair = tuple(sorted([entity1, entity2]))  # Sort pair for consistent key ordering
                    cooccurrence_frequency[pair] = cooccurrence_frequency.get(pair, 0) + 1

    # Now we create the graph by adding nodes and edges based on the co-occurrence frequency
    for entity, frequency in entity_frequency.items():
        G.add_node(entity, frequency=frequency)  # Add node with entity frequency as an attribute

    # Add edges between co-occurring entities
    for (entity1, entity2), cooccurrence_count in cooccurrence_frequency.items():
        # Calculate the weight: (co-occurrence frequency of E1 and E2) / max(frequency of E1, frequency of E2)
        weight = cooccurrence_count / max(entity_frequency[entity1], entity_frequency[entity2])
        # Add the edge between the entities with the calculated weight
        G.add_edge(entity1, entity2, weight=weight, cooccurrence_count=cooccurrence_count)

    return G, word_segmented_sentences

# Function to visualize the knowledge graph
def visualize_knowledge_graph(G):
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(10, 7))

    # Draw the graph with nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()

# Function to print entities and co-occurrences
def print_entities_and_cooccurrences(G):
    print("Entities:")
    for node in G.nodes(data=True):
        print(f"Entity: {node[0]}, Frequency: {node[1]['frequency']}")

    print("\nCo-occurrences (Edges):")
    for edge in G.edges(data=True):
        print(f"Entity 1: {edge[0]}, Entity 2: {edge[1]}, Co-occurrence Count: {edge[2]['cooccurrence_count']}, Weight: {edge[2]['weight']}")

# Function to print word-segmented sentences
def print_word_segmented_sentences(sentences):
    print("\nWord-Segmented Sentences:")
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: {sentence}")

def convert_networkx_to_data(G):
    data = from_networkx(G)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    return data
# Example usage
if __name__ == "__main__":
    # Example text provided in your input
    text = "Sau hai thế kỷ, chính sách đóng cửa đất nước dưới thời Mạc phủ Tokugawa đã đi đến kết thúc khi Nhật Bản bị Hoa Kỳ ép mở cửa giao thương vào năm 1854. Những năm tiếp theo cuộc Minh Trị duy tân năm 1868 và sự sụp đổ của chế độ mạc phủ, Nhật Bản đã tự chuyển đổi từ một xã hội khá lạc hậu và phong kiến sang một quốc gia công nghiệp hiện đại. Nhật đã cử các phái đoàn và sinh viên đi khắp thế giới để học và tiếp thu khoa học và nghệ thuật phương Tây, điều này đã được thực hiện nhằm giúp Nhật Bản tránh khỏi rơi vào ách thống trị của nước ngoài và cũng giúp cho Nhật có thể cạnh tranh ngang ngửa với các cường quốc phương Tây."
        
    # Build the knowledge graph from text and get word-segmented sentences
    G, word_segmented_sentences = build_knowledge_graph(text)

    # Print entities and co-occurrences
    print_entities_and_cooccurrences(G)

    # Print word-segmented sentences
    print_word_segmented_sentences(word_segmented_sentences)

    visualize_knowledge_graph(G)

    print(G)
