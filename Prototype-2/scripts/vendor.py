from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
import os

# Download the punkt tokenizer models (only downloads once)
nltk.download('punkt_tab', quiet=True)

# Resolve model path relative to THIS file, not the CWD — so Node.exe can be run from any folder
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH  = os.path.join(_SCRIPT_DIR, '..', 'models', 'local_minilm')

# Load the model globally so it stays loaded in memory until the program stops
model = SentenceTransformer(_MODEL_PATH)

sentences = ["The cat sits outside", "A man is playing guitar"]

def get_embedding(text : str):
    # Deterministic int8: get normalized floats, multiply by 127, cast to int8
    emb = model.encode(text, normalize_embeddings=True)
    return np.clip(emb * 127.0, -128, 127).astype(np.int8)

def get_embeddings(texts: list[str]):
    """
    Returns a list of 384-dimensional deterministic int8 embeddings.
    """
    if isinstance(texts, str):
        texts = [texts]
    emb = model.encode(texts, normalize_embeddings=True)
    return np.clip(emb * 127.0, -128, 127).astype(np.int8)

def chunk_file(filepath: str) -> list[str]:
    """
    Reads a .txt file and splits it into sentence-level chunks using NLTK.
    Returns a list of sentence strings.
    """
    if not os.path.exists(filepath):
        print(f"[vendor.py] Warning: File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if not text.strip():
        return []
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    # Filter out empty/whitespace-only sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def chunk_file_with_offsets(filepath: str) -> list[tuple[str, int, int]]:
    """
    Reads a .txt file, splits into sentences, and finds their exact byte offset
    and byte length in the original file.
    Returns a list of tuples: (sentence_text, byte_offset, byte_length)
    """
    if not os.path.exists(filepath):
        print(f"[vendor.py] Warning: File not found: {filepath}")
        return []
        
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        text = f.read()
        
    if not text.strip():
        return []
        
    sentences = nltk.sent_tokenize(text)
    
    results = []
    current_char_idx = 0
    
    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped:
            continue
            
        start_char_idx = text.find(s_stripped, current_char_idx)
        if start_char_idx == -1:
            start_char_idx = current_char_idx # Fallback
            
        current_char_idx = start_char_idx + len(s_stripped)
        
        # Calculate byte offsets
        byte_offset = len(text[:start_char_idx].encode('utf-8'))
        byte_length = len(s_stripped.encode('utf-8'))
        
        results.append((s_stripped, byte_offset, byte_length))
        
    return results

def embed_chunks(sentences: list[str]) -> list:
    """
    Batch-embeds a list of sentences.
    Returns a list of 384-dimensional deterministic int8 embeddings as Python lists.
    """
    if not sentences:
        return []
    
    emb = model.encode(sentences, normalize_embeddings=True)
    embeddings = np.clip(emb * 127.0, -128, 127).astype(np.int8)
    # Convert numpy array to list of lists for easy C++ consumption
    return embeddings.tolist()

def main():
    print("Generating embeddings for sentences...")
    embeddings = get_embeddings(sentences)
    print(f"Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else len(embeddings)}")
    print("Embeddings array:", embeddings)

if __name__ == "__main__":
    main()
