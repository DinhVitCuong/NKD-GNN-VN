import py_vncorenlp
from word_lists import PEOPLE, PLACE, ORGANIZATION, number_map
# Automatically download VnCoreNLP components from the original repository
py_vncorenlp.download_model(save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

# Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models`
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner"], save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

PERSON_PLACEHOLDER = "<PERSON>"
PLACE_PLACEHOLDER = "<PLACE>"
ORGANIZATION_PLACEHOLDER = "<ORGANIZATION>"
def generate_template_caption(text, model):

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
            #To reduce from 2-3 posTag "N" sit next together only return 1 "N" tag
            flag = False
            # Process each token in the sentence
            prev_word = ""
            prev_pos_tag = ""
            for token in sentence:
                word = token['wordForm']
                pos_tag = token['posTag']
                if pos_tag == 'N' and flag == False:
                    if word.lower() in PEOPLE:
                        if prev_pos_tag == 'M':
                            try:
                                num_person = number_map[prev_word.lower()]
                            except:
                                num_person = 1
                            result.pop()
                            for i in range(num_person - 1):
                                result.append(PERSON_PLACEHOLDER + ",")
                        result.append(PERSON_PLACEHOLDER)
                    elif word.lower() in PLACE:
                        result.append(PLACE_PLACEHOLDER)
                    elif word.lower() in ORGANIZATION:
                        result.append(ORGANIZATION_PLACEHOLDER)
                    else:
                        result.append(word)
                    flag = True
                elif pos_tag == 'N' and flag == True:
                    continue
                elif pos_tag !='Nc' and pos_tag != 'P':
                    result.append(word)
                    flag = False
                    prev_word = word
                    prev_pos_tag = pos_tag

        caption.append(' '.join(result))
    # Join the processed tokens back into a sentence
    return '. '.join(caption).strip(" .")
####EXAMPLE USAGE:
# text = "Anh ấy đã phát biểu tại tòa nhà Quốc hội. Ba người đàn ông đang đấm nhau ở quảng trường. Tiểu đội 3 đang tiến quân tới sân vận động"
# print(f'Original: {text}')
# print(f'Template: {generate_template_caption(text)}')