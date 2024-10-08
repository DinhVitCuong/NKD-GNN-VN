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

# Function to generate template captions with placeholders
def generate_template_caption(text):
    # Annotate the text with VnCoreNLP to get named entities (with wseg and ner)
    annotations = model.annotate_text(text)

    # Initialize a list to store the processed tokens
    result = []

    # Variables to handle multi-token entities
    current_entity = None
    current_placeholder = None

    # Loop over each sentence in the annotation result (annotations is a dictionary with indices as keys)
    for sentence_key in annotations:
        sentence = annotations[sentence_key]
        
        # Process each token in the sentence
        for token in sentence:
            word = token['wordForm']
            ner_tag = token['nerLabel']  # Get NER label from the token

            # Check for beginning of a new entity
            if ner_tag.startswith('B-'):
                # If there was a previous entity, append its placeholder
                if current_entity:
                    result.append(current_placeholder)
                # Set new entity
                current_entity = ner_tag[2:]  # Get the entity type (e.g., PER, LOC, ORG)
                if current_entity == 'PER':
                    current_placeholder = PERSON_PLACEHOLDER
                elif current_entity == 'LOC':
                    current_placeholder = PLACE_PLACEHOLDER
                elif current_entity == 'ORG':
                    current_placeholder = ORGANIZATION_PLACEHOLDER
                else:
                    current_placeholder = get_building_placeholder(word)  # Handle building-related words
            elif ner_tag.startswith('I-'):
                # Continue the current entity, do nothing (placeholder already set)
                continue
            else:
                # If we're done with an entity, append the placeholder and reset
                if current_entity:
                    result.append(current_placeholder)
                    current_entity = None
                    current_placeholder = None
                # If not an entity, append the word (could be normal words or punctuation)
                result.append(get_building_placeholder(word))

        # If the sentence ends and we still have an active entity, append the placeholder
        if current_entity:
            result.append(current_placeholder)
            current_entity = None
            current_placeholder = None

    # Join the processed tokens back into a sentence
    return ' '.join(result)

# Example usage
if __name__ == "__main__":
    # Sample Vietnamese text containing named entities and building-related words
    vietnamese_sentence = "Đinh Việt Cường đã phát biểu tại tòa nhà Quốc hội."

    # Generate template caption with placeholders
    template_caption = generate_template_caption(vietnamese_sentence)

    print("Original Sentence: ", vietnamese_sentence)
    print("Template Caption: ", template_caption)
