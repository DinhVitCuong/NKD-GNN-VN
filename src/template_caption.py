import stanza
from word_lists import PEOPLE, PLACE, ORGANIZATION, number_map

# Tải model tiếng Việt
stanza.download('vi')

# Khởi tạo pipeline với tokenize, pos, ner
nlp = stanza.Pipeline('vi', processors='tokenize,pos,ner')

def generate_template_caption(text, nlp):
    PERSON_PLACEHOLDER = "<PERSON>"
    PLACE_PLACEHOLDER = "<PLACE>"
    ORGANIZATION_PLACEHOLDER = "<ORGANIZATION>"
    
    sentences = text.split(".")
    caption = []
    for sent in sentences:
        doc = nlp(sent)
        result = []
        i = 0
        doc = doc.sentences[0]
        entities = doc.entities
        
        while i < len(doc.words):
            word = doc.words[i]
            text = word.text
            pos_tag = word.upos
            
            # Xử lý entity (PER/LOC/ORG)
            entity_found = False
            for entity in entities:
                if word.start_char >= entity.start_char and word.end_char <= entity.end_char:
                    if entity.type == 'PERSON':
                        result.append(PERSON_PLACEHOLDER)
                    elif entity.type == 'LOCATION':
                        result.append(PLACE_PLACEHOLDER)
                    elif entity.type == 'ORGANIZATION':
                        result.append(ORGANIZATION_PLACEHOLDER)
                    while word.end_char != entity.end_char:
                        i += 1
                        word = doc.words[i]
                    i += 1
                    entity_found = True
                    break
            if entity_found:
                continue
            
            # Xử lý số lượng + danh từ
            if pos_tag == 'NUM' and (i + 1 < len(doc.words)):
                next_word = doc.words[i + 1]
                if next_word.text.lower() in PEOPLE:
                    num = number_map.get(text.lower(), 1)
                    result.extend([PERSON_PLACEHOLDER] * num)
                    i += 2
                    while i < len(doc.words) and doc.words[i].upos in ['NOUN', 'PROPN', 'PRON']:
                        i += 1
                    continue
            
            # Xử lý danh từ: chỉ lấy danh từ đầu tiên
            if pos_tag in ['NOUN', 'PROPN']:
                if text.lower() in PEOPLE:
                    result.append(PERSON_PLACEHOLDER)
                elif text.lower() in PLACE:
                    result.append(PLACE_PLACEHOLDER)
                elif text.lower() in ORGANIZATION:
                    result.append(ORGANIZATION_PLACEHOLDER)
                else:
                    result.append(text)
                # Bỏ qua danh từ sau (nếu có)
                if i + 1 < len(doc.words) and doc.words[i + 1].upos in ['NOUN', 'PROPN', 'PRON']:
                    i += 1
            else:
                # Xử lý dấu câu
                if pos_tag == 'PUNCT' and result:
                    result[-1] += text
                else:
                    result.append(text)
            
            i += 1
        
        caption.append(' '.join(result))
    
    return '. '.join(caption).strip(" .").strip(".")

# text = "Anh ấy đã phát biểu tại tòa nhà Quốc hội. Ba người đàn ông đang đấm nhau ở quảng trường. Tiểu đội 3 đang tiến quân tới sân vận động"
# print(f'Original: {text}')
# print(f'Template: {generate_template_caption(text, nlp)}')

