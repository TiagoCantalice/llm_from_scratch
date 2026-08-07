import re

# Ouvrir le texte
with open(r"C:\Users\canta\Google Drive\llm_from_scratch\llm_from_scratch\chapter_2\the-verdict.txt", "r", encoding='utf-8') as f:
    text = f.read()

# Preprocessing le text. Ici On a divisé le texte dans morceaux à partir des variables
preprocessed = re.split(r'([,./:;?_!?"()]|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed]

### Ensuite, on a crée le vocabulaire

# Il faut prendre les uniques mots du texte
all_tokens = sorted(set(preprocessed))

# Nous irons ajouter le "unk" (Quand on n'a pas mot dans le vocabulaire) 
# et "endoftext" (Pour distinguer entre deux textes pendant l'entrainement)

all_tokens.extend(["<|endoftext|>", "<|unk|>"])

print(all_tokens)

# Créer le vocabulaire
vocab = {token: integer for integer, token in enumerate(all_tokens)} 

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# Utiliser le SimpleTokenizer version deux

from SimpleTokenizer import SimpleTokenizerV2

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# Instancier la classe
tokenizer = SimpleTokenizerV2(vocab)

# Encode le texte
print(tokenizer.encode(text))

# Decode le texte
print(tokenizer.decode(tokenizer.encode(text)))



