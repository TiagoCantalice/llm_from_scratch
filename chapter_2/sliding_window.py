with open(r"C:\Users\canta\Google Drive\llm_from_scratch\llm_from_scratch\chapter_2\the-verdict.txt", "r", encoding='utf-8') as f:
    text = f.read()
#print("Numéro de caractères dans le texte: ", len(text))
#print(text[:99])  # Affiche les 99 premiers caractères du texte

import re
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(text)
print(len(enc_text))

# Removing the first 50 tokens from the encoded text
enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]

print("Encoded sample: ", x)

y = enc_sample[1:context_size+1]
print("Shifted sample: ", y)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i:i+1]
    print(f"Context: {context}, -------> Desired: {desired}")    

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i:i+1]
    print(f"{tokenizer.decode(context)}, -------> {tokenizer.decode(desired)}")    

