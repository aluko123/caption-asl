import re

with open('keywords.txt', 'r') as f:
    asl_data = f.readlines()

asl_words = {line.split('"')[3].upper() for line in asl_data if line.strip()}


def generate_asl_gloss(sentence):
    words = re.findall(r'\b\w+\b', sentence.lower())
    gloss = [word.upper() for word in words if word.upper() in asl_words]
    return ' '.join(gloss)

with open('word_list2.txt', 'r') as f:
    sentences = [line.strip() for line in f if line.strip()]

asl_sentences = [(sentence, generate_asl_gloss(sentence)) for sentence in sentences]

with open('translated_text2.txt', 'w') as f:
    for data in asl_sentences:
        f.write(f"{data}\n")

for english, asl in asl_sentences:
    print(f"('{english}', '{asl}')")