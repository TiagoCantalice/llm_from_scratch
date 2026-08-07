import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        " of someunknownPlace.")

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# The tokenizer BEP (Byte pair encoding) breaks down unkwokn words up to the point the 
# subwords are known. For example, the word "someunknownPlace" is broken down 
# into the following subwords: "some", "unknown", "Place". The tokenizer will 
# then encode each of these subwords into their corresponding integer IDs. 

##  Exercise 2.1

word_text = "Akwirw ier"

word_integers = tokenizer.encode(word_text, allowed_special={"<|endoftext|>"})
print("Encoded IDs for the word_text: ", word_integers)
print(word_integers)

#Checking the vocab of the tokenizer
vocab = {tokenizer.decode([i]): i for i in range(tokenizer.n_vocab) if tokenizer.decode([i]) in [tokenizer.decode([j]) for j in word_integers]}

print("The vocab of the tokenizer: ", vocab)

# Getting the decoded text for the word_text

print("Decoded text for the word_text: ", tokenizer.decode(word_integers))