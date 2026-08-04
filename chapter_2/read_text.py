with open(r"C:\Users\canta\Google Drive\llm_from_scratch\llm_from_scratch\chapter_2\the-verdict.txt", "r", encoding='utf-8') as f:
    text = f.read()
#print("Numéro de caractères dans le texte: ", len(text))
#print(text[:99])  # Affiche les 99 premiers caractères du texte

import re

#text_check = "Hello, world. This, is a text."

#result = re.split(r'([,.]|\s)', text_check)
#result = [item for item in result if item.strip()]
#print(result)

#test2 = "Hello, world. Is this-- a test?"

#result2 = re.split(r'([,./:;?_!?"()]|--|\s)', test2)
#result2 = [item.strip() for item in result2 if item.strip()]
#print(result2)

#----------------------

result_text = re.split(r'([,./:;?_!?"()]|--|\s)', text)
result_text = [item.strip() for item in result_text if item.strip()]
#print(result_text[:30])

all_words = sorted(set(result_text))

#print("la taille du vocabulaire: ", len(all_words))

vocab = {word: i for i, word in enumerate(all_words)}

"""for i, word in enumerate(vocab.items()):
    print(word)
    if i >= 50:
        break"""

# Using the SimpleTokenizerV1 class to encode and decode the text

from SimpleTokenizer import SimpleTokenizerV1

tokenizer = SimpleTokenizerV1(vocab)
text_example = """It's the last he painted"""
ids = tokenizer.encode(text_example)
print("Encoded IDs: ", ids)

decoded_text = tokenizer.decode(ids)
print("Decoded text: ", decoded_text)
