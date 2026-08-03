import nltk

# Download the punkt tokenizer models
nltk.download('punkt_tab')

text = "Hello world! This is a simple sentence. We can break text apart using NLTK."

# Split the text into sentences
sentences = nltk.sent_tokenize(text)

for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1}: {sentence}")
