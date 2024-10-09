import py_vncorenlp
from word_lists import PEOPLE, PLACE, ORGANIZATION
# Automatically download VnCoreNLP components from the original repository
py_vncorenlp.download_model(save_dir=r'E:\DATN\NKD-GNN-test\VnCoreNLP')

# Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models`
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner"], save_dir=r'E:\DATN\NKD-GNN-test\VnCoreNLP')

# Define placeholder tags
PERSON_PLACEHOLDER = "<PERSON>"
PLACE_PLACEHOLDER = "<PLACE>"
ORGANIZATION_PLACEHOLDER = "<ORGANIZATION>"

# # Define a list of common words in Vietnamese


# Function to generate template captions with placeholders
def generate_template_caption(text):

    # Split into sentences
    sentences = text.split(".")

    # Annotate the chunk with VnCoreNLP
    caption = []
    for splited_sentence in sentences:
        annotations = model.annotate_text(splited_sentence)

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
                pos_tag = token['posTag']
                if pos_tag == 'N':
                    if word in PLACE:
                        result.append(PLACE_PLACEHOLDER)
                    elif word in PEOPLE:
                        result.append(PERSON_PLACEHOLDER)
                    elif word in ORGANIZATION:
                        result.append(ORGANIZATION_PLACEHOLDER)
                    else:
                        result.append(word)
                else:
                    result.append(word)
        caption = caption.append(' '.join(result))
    # Join the processed tokens back into a sentence
    return ' '.join(caption)


# Example usage
if __name__ == "__main__":
    # Sample Vietnamese text containing named entities and building-related words
    vietnamese_sentence = "Anh ấy đã phát biểu tại tòa nhà Quốc hội. Hai người đàn ông đang đấm nhau ở quảng trường"

    # Generate template caption with placeholders
    template_caption = generate_template_caption(vietnamese_sentence)

    print("Original Sentence: ", vietnamese_sentence)
    print("Template Caption: ", template_caption)
