import stanza
from word_lists import PEOPLE, PLACE, ORGANIZATION, number_map

# Download and load the Vietnamese model for Stanza
stanza.download('vi')
model = stanza.Pipeline(lang='vi', processors='tokenize,pos,ner')

PERSON_PLACEHOLDER = "<PERSON>"
PLACE_PLACEHOLDER = "<PLACE>"
ORGANIZATION_PLACEHOLDER = "<ORGANIZATION>"

def generate_template_caption(text, model):

    # Split into sentences
    sentences = text.split(".")

    caption = []
    for splited_sentence in sentences:
        doc = model(splited_sentence.strip())

        # Initialize a list to store the processed tokens
        result = []

        # Variables to handle multi-token entities
        flag = False
        prev_word = ""
        prev_pos_tag = ""

        # Process each sentence in the doc
        for sentence in doc.sentences:
            for word in sentence.words:
                word_text = word.text
                pos_tag = word.xpos  # Use 'xpos' for Vietnamese POS tags

                # For NER-based placeholders
                if word_text in PEOPLE:
                    result.append(PERSON_PLACEHOLDER)
                elif word_text in PLACE:
                    result.append(PLACE_PLACEHOLDER)
                elif word_text in ORGANIZATION:
                    result.append(ORGANIZATION_PLACEHOLDER)
                # If the POS tag is 'N' and no previous 'N' tag was processed, add the token
                elif pos_tag == 'N' and not flag:
                    result.append(word_text)
                    flag = True
                # Ignore consecutive 'N' tags
                elif pos_tag == 'N' and flag:
                    continue
                # For other cases, reset flag and process the token normally
                else:
                    result.append(word_text)
                    flag = False

                # Track previous word and POS tag for number handling
                prev_word = word_text
                prev_pos_tag = pos_tag

        caption.append(' '.join(result))
    # Join the processed tokens back into a sentence
    return '. '.join(caption).strip(" .")

#### EXAMPLE USAGE:
# text = "Anh ấy đã phát biểu tại tòa nhà Quốc hội. Ba người đàn ông đang đấm nhau ở quảng trường. Tiểu đội 3 đang tiến quân tới sân vận động"
# print(f'Original: {text}')
# print(f'Template: {generate_template_caption(text, model)}')
