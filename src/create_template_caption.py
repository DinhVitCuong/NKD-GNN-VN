from vncorenlp import VnCoreNLP
import os

VNLPCORENLP_PATH = r'E:\DATN\NKD-GNN-test\VnCoreNLP\VnCoreNLP-1.2.jar'

# Start the VnCoreNLP server
rdrsegmenter = VnCoreNLP(VNLPCORENLP_PATH, annotators="wseg,pos,ner", max_heap_size='-Xmx2g')

# Define placeholder tags
PERSON_PLACEHOLDER = "<Person>"
PLACE_PLACEHOLDER = "<Place>"
ORGANIZATION_PLACEHOLDER = "<Organization>"
BUILDING_PLACEHOLDER = "<Building>"

# Define a list of common building-related words in Vietnamese
building_keywords = ['nhà', 'tòa nhà', 'trường học', 'bệnh viện', 'khách sạn', 'sân bay', 'siêu thị', 'bảo tàng', 'trụ sở', 'chung cư']

# Function to detect and replace building-related words
def get_building_placeholder(word):
    if word.lower() in building_keywords:
        return BUILDING_PLACEHOLDER
    return word

# Function to generate template captions with placeholders
def generate_template_caption_vietnamese_with_buildings(text):
    # Annotate the text with VnCoreNLP to get named entities and POS tags
    annotations = rdrsegmenter.annotate(text)

    # Initialize a list to store the processed tokens
    result = []

    # Loop over each sentence in the annotation result
    for sentence in annotations['sentences']:
        # Process each token in the sentence
        for token in sentence:
            word = token['form']
            ner_tag = token['nerLabel']

            # Replace named entities with placeholders
            if ner_tag == 'B-PER' or ner_tag == 'I-PER':
                result.append(PERSON_PLACEHOLDER)
            elif ner_tag == 'B-LOC' or ner_tag == 'I-LOC':
                result.append(PLACE_PLACEHOLDER)
            elif ner_tag == 'B-ORG' or ner_tag == 'I-ORG':
                result.append(ORGANIZATION_PLACEHOLDER)
            else:
                # Replace building-related words with the building placeholder
                result.append(get_building_placeholder(word))

    # Join the processed tokens back into a sentence
    return ' '.join(result)

# Example usage
if __name__ == "__main__":
    # Sample Vietnamese text containing named entities and building-related words
    vietnamese_sentence = "Nguyễn Phú Trọng đã phát biểu tại tòa nhà Quốc hội và trường Đại học Quốc gia Hà Nội."

    # Generate template caption with placeholders
    template_caption = generate_template_caption_vietnamese_with_buildings(vietnamese_sentence)

    print("Original Sentence: ", vietnamese_sentence)
    print("Template Caption: ", template_caption)
