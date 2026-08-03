from sentence_transformers import SentenceTransformer

print("Downloading and saving model locally...")
# This pulls it from the internet one last time
model = SentenceTransformer('all-MiniLM-L6-v2')

# This saves the actual model weights and config files into a local folder
model.save('./local_minilm')
print("Done! The model is now saved in the './local_minilm' folder.")