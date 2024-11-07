import networkx as nx
import stanza
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx

# Download and load the Vietnamese model for Stanza
stanza.download('vi')
model = stanza.Pipeline(lang='vi', processors='tokenize,ner')

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

    # Annotate each sentence with Stanza
    for splited_sentence in sentences:
        doc = model(splited_sentence.strip())

        for sentence in doc.sentences:
            # Store the word-segmented sentence
            word_segmented_sentence = ' '.join([word.text for word in sentence.words])
            word_segmented_sentences.append(word_segmented_sentence)

            # Initialize a list for the current sentence's entities
            entities_in_sentence = []
            
            # Loop through each entity in the sentence
            for entity in sentence.ents:
                entity_text = entity.text.replace(" ", "_")  # Replace spaces with underscores
                entity_type = entity.type

                # Add entity to the sentence's entity list
                entities_in_sentence.append((entity_text, entity_type))

                # Track frequency of the entity
                entity_frequency[entity_text] = entity_frequency.get(entity_text, 0) + 1

            # Track co-occurrence for entities in this sentence
            for i, (entity1, type1) in enumerate(entities_in_sentence):
                for entity2, type2 in entities_in_sentence[i + 1:]:
                    pair = tuple(sorted([entity1, entity2]))  # Sort pair for consistent key ordering
                    cooccurrence_frequency[pair] = cooccurrence_frequency.get(pair, 0) + 1

    # Create the graph by adding nodes and edges based on co-occurrence frequency
    for entity, frequency in entity_frequency.items():
        G.add_node(entity, frequency=frequency)

    for (entity1, entity2), cooccurrence_count in cooccurrence_frequency.items():
        weight = cooccurrence_count / max(entity_frequency[entity1], entity_frequency[entity2])
        G.add_edge(entity1, entity2, weight=weight, cooccurrence_count=cooccurrence_count)

    return G, word_segmented_sentences

# Function to visualize the knowledge graph
def visualize_knowledge_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
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

    # Visualize the knowledge graph
    visualize_knowledge_graph(G)
