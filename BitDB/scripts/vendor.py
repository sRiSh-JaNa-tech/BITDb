import os
import sys
import numpy as np
import nltk

# Check punkt tokenizer locally first to avoid network timeout
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

# Resolve model paths relative to THIS file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
_MODEL_PATH = os.path.abspath(os.path.join(_PROJECT_ROOT, 'models', 'local_minilm'))
_OV_MODEL_PATH = os.path.abspath(os.path.join(_PROJECT_ROOT, 'models', 'openvino_minilm'))

_HAS_LOCAL_MINILM = os.path.isdir(_MODEL_PATH) and os.path.isfile(os.path.join(_MODEL_PATH, "config.json"))
_MODEL_NAME_OR_PATH = _MODEL_PATH if _HAS_LOCAL_MINILM else "sentence-transformers/all-MiniLM-L6-v2"

backend_type = "cpu"
ov_model = None
ov_tokenizer = None
st_model = None

# 1. Try Intel OpenVINO on iGPU (OpenCL / Level Zero)
try:
    import openvino as ov
    core = ov.Core()
    if "GPU" in core.available_devices and os.path.exists(_OV_MODEL_PATH):
        from transformers import AutoTokenizer
        from optimum.intel.openvino import OVModelForFeatureExtraction
        
        ov_tokenizer = AutoTokenizer.from_pretrained(_OV_MODEL_PATH, local_files_only=True)
        ov_model = OVModelForFeatureExtraction.from_pretrained(_OV_MODEL_PATH, export=False, device="GPU")
        backend_type = "openvino_gpu"
        gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
        print(f"[vendor.py] Hardware Acceleration: OpenVINO iGPU ({gpu_name}) active.", flush=True)
except Exception:
    pass

# 2. Fallback to CUDA Discrete GPU or standard CPU
if backend_type == "cpu":
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(_MODEL_NAME_OR_PATH, device=device, local_files_only=_HAS_LOCAL_MINILM)
    st_model.eval()
    for param in st_model.parameters():
        param.requires_grad = False
    backend_type = device
    print(f"[vendor.py] Hardware Acceleration: Running on {device.upper()} (Weights Frozen, Eval Mode).", flush=True)

# ─────────────────────────────────────────────────────────────────────
# Embedding functions
# ─────────────────────────────────────────────────────────────────────

def _encode_batch_np(sentences: list) -> np.ndarray:
    if not sentences:
        return np.zeros((0, 384), dtype=np.int8)
    
    if backend_type == "openvino_gpu" and ov_model is not None:
        inputs = ov_tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="np")
        outputs = ov_model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        mask = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden * mask, axis=1)
        sum_mask = np.clip(mask.sum(axis=1), 1e-9, None)
        mean_embeddings = sum_embeddings / sum_mask
        
        norms = np.linalg.norm(mean_embeddings, ord=2, axis=1, keepdims=True)
        normalized = mean_embeddings / np.clip(norms, 1e-12, None)
        return np.clip(normalized * 127.0, -128, 127).astype(np.int8)
    else:
        emb = st_model.encode(sentences, normalize_embeddings=True)
        return np.clip(emb * 127.0, -128, 127).astype(np.int8)

def get_embedding(text: str):
    emb = _encode_batch_np([text])
    return emb[0]

def get_embeddings(texts: list):
    if isinstance(texts, str):
        texts = [texts]
    return _encode_batch_np(texts)

def embed_chunks(sentences: list) -> list:
    if not sentences:
        return []
    embeddings = _encode_batch_np(sentences)
    return embeddings.tolist()

# ─────────────────────────────────────────────────────────────────────
# Text file chunking (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────

def chunk_file(filepath: str) -> list:
    """Reads a .txt file and splits into sentence-level chunks via NLTK."""
    if not os.path.exists(filepath):
        print(f"[vendor.py] Warning: File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    if not text.strip():
        return []
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_file_with_offsets(filepath: str) -> list:
    """
    Reads a .txt file, splits into sentences, and returns
    (sentence_text, byte_offset, byte_length) tuples.
    """
    if not os.path.exists(filepath):
        print(f"[vendor.py] Warning: File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        text = f.read()
    if not text.strip():
        return []
    return _chunk_text_to_offsets(text)

# ─────────────────────────────────────────────────────────────────────
# Raw text chunking — Prototype-3 addition
# ─────────────────────────────────────────────────────────────────────

def _chunk_text_to_offsets(text: str) -> list:
    """
    Internal helper: chunks a text string into sentences and returns
    (sentence_text, byte_offset, byte_length) tuples.
    """
    import re
    # Clean mid-sentence line breaks from raw PDF extraction so they don't break tokenization.
    text_clean = re.sub(r'(?<![.!?])\s*\n\s*', ' ', text)
    
    sentences = nltk.sent_tokenize(text_clean)
    
    refined_sentences = []
    for s in sentences:
        s = s.strip()
        # Filter out very short fragments (like isolated headings, numbers, or 2-word artifacts)
        # to ensure only rich, complete sentences are indexed.
        if len(s) > 30 and len(s.split()) >= 5:
            refined_sentences.append(s)
            
    # Group into overlapping windows (e.g. 3 sentences per chunk, overlap by 1 sentence)
    window_size = 3
    stride = 2
    windowed_chunks = []
    
    i = 0
    while i < len(refined_sentences):
        window = refined_sentences[i : i + window_size]
        windowed_chunks.append(" ".join(window))
        if i + window_size >= len(refined_sentences):
            break
        i += stride
            
    results = []
    for chunk_text in windowed_chunks:
        # We just need to return the string and its encoded length;
        # Prototype-4 Build.cpp calculates its own contiguous text offsets.
        byte_length = len(chunk_text.encode('utf-8'))
        results.append((chunk_text, 0, byte_length))
    return results

def chunk_text_with_offsets(text: str) -> list:
    """
    Prototype-3: Chunks a raw text string (not a file path) into sentences.
    Returns (sentence_text, byte_offset, byte_length) tuples.
    Used for per-page PDF text already extracted in memory.
    """
    if not text or not text.strip():
        return []
    return _chunk_text_to_offsets(text)

# ─────────────────────────────────────────────────────────────────────
# PDF extraction — Prototype-3 addition
# ─────────────────────────────────────────────────────────────────────

def pdf_to_text_pages(filepath: str) -> list:
    """
    Prototype-3: Extracts text from a PDF file page by page.
    Returns a list of (page_num, page_text) tuples (0-indexed page_num).
    """
    try:
        from pdf_extractor import pdf_to_text_pages as _extract
        return _extract(filepath)
    except ImportError as e:
        print(f"[vendor.py] ERROR: pdf_extractor not available: {e}")
        return []
    except Exception as e:
        print(f"[vendor.py] ERROR in pdf_to_text_pages: {e}")
        return []
