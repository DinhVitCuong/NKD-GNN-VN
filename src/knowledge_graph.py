import networkx as nx
import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
py_vncorenlp.download_model(save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

# Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models`
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "ner"], save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

# Function to extract named entities and build the knowledge graph
def build_knowledge_graph(text):
    # Annotate the text with VnCoreNLP to get word-segmented sentences and named entities
    annotations = model.annotate_text(text)

    # Create an empty graph
    G = nx.Graph()

    # Initialize list to store the segmented sentences
    word_segmented_sentences = []

    # Collect all unique entities across the entire text and add them as nodes to G
    entities = set()

    # Loop through each sentence and extract entities
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
                    current_entity = None
                    current_type = None

        # Add the last entity in the sentence if it exists
        if current_entity:
            entities_in_sentence.append((current_entity.strip(), current_type))

        # Add entities from this sentence to the global entity set (to add as nodes to G later)
        for entity, entity_type in entities_in_sentence:
            entities.add((entity, entity_type))

    # Add nodes (entities) to the graph
    for entity, entity_type in entities:
        G.add_node(entity, entity_type=entity_type)

    # Second loop: Process sentences again to add edges between co-occurring entities
    for sentence_key in annotations:
        sentence = annotations[sentence_key]

        # Extract entities from the current sentence
        entities_in_sentence = []
        current_entity = None
        current_type = None

        for token in sentence:
            word = token['wordForm']
            ner_tag = token['nerLabel']

            if ner_tag.startswith('B-'):
                if current_entity:
                    entities_in_sentence.append((current_entity.strip(), current_type))
                current_entity = word
                current_type = ner_tag[2:]
            elif ner_tag.startswith('I-') and current_entity:
                current_entity += f"_{word}"
            else:
                if current_entity:
                    entities_in_sentence.append((current_entity.strip(), current_type))
                    current_entity = None
                    current_type = None

        if current_entity:
            entities_in_sentence.append((current_entity.strip(), current_type))

        # Add edges between co-occurring entities within the same sentence
        for i, (entity1, type1) in enumerate(entities_in_sentence):
            for entity2, type2 in entities_in_sentence[i + 1:]:
                if G.has_edge(entity1, entity2):
                    # Increment weight if edge already exists
                    G[entity1][entity2]['weight'] += 1
                else:
                    # Add edge with weight = 1
                    G.add_edge(entity1, entity2, weight=1)

    return G, word_segmented_sentences

# Function to print entities and co-occurrences
def print_entities_and_cooccurrences(G):
    print("Entities:")
    for node in G.nodes(data=True):
        print(f"Entity: {node[0]}, Type: {node[1]['entity_type']}")

    print("\nCo-occurrences (Edges):")
    for edge in G.edges(data=True):
        print(f"Entity 1: {edge[0]}, Entity 2: {edge[1]}, Co-occurrence Count: {edge[2]['weight']}")

# Function to print word-segmented sentences
def print_word_segmented_sentences(sentences):
    print("\nWord-Segmented Sentences:")
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: {sentence}")

# Example usage
if __name__ == "__main__":
    # Example text provided in your input
    text = "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây. Ông Nguyễn Khắc Chúc đang nói chuyện với giám đốc ký túc xá."

    # Build the knowledge graph from text and get word-segmented sentences
    G, word_segmented_sentences = build_knowledge_graph(text)

    # Print entities and co-occurrences
    print_entities_and_cooccurrences(G)

    # Print word-segmented sentences
    print_word_segmented_sentences(word_segmented_sentences)
