import os
import sys
import numpy as np
import nltk

# Ensure fast_pdf_agent is discoverable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
_p = os.path.join(_PROJECT_ROOT, 'fast_pdf_agent')
if os.path.exists(_p) and _p not in sys.path:
    sys.path.insert(0, _p)

# Check punkt tokenizer locally first to avoid network timeout
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

# Resolve model paths relative to THIS file
_MODEL_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'models', 'local_minilm'))
_OV_MODEL_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'models', 'openvino_minilm'))

backend_type = "cpu"
ov_model = None
ov_tokenizer = None
st_model = None

# 1. Try Intel OpenVINO on iGPU (OpenCL / Level Zero)
try:
    import openvino as ov
    core = ov.Core()
    if "GPU" in core.available_devices:
        from transformers import AutoTokenizer
        from optimum.intel.openvino import OVModelForFeatureExtraction
        
        load_path = _OV_MODEL_PATH if os.path.exists(_OV_MODEL_PATH) else _MODEL_PATH
        ov_tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
        ov_model = OVModelForFeatureExtraction.from_pretrained(load_path, device="GPU", local_files_only=True)
        backend_type = "openvino_gpu"
        gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
        print(f"[vendor.py] Hardware Acceleration: OpenVINO iGPU ({gpu_name}) active.", flush=True)
except Exception as e:
    print(f"[vendor.py] OpenVINO iGPU notice: {e}", flush=True)

# 2. Fallback to CUDA Discrete GPU or standard CPU
if backend_type == "cpu":
    import torch
    from sentence_transformers import SentenceTransformer
    if torch.cuda.is_available():
        st_model = SentenceTransformer(_MODEL_PATH, device="cuda", local_files_only=True)
        backend_type = "cuda"
        print("[vendor.py] Hardware Acceleration: Discrete GPU (CUDA) detected.", flush=True)
    else:
        st_model = SentenceTransformer(_MODEL_PATH, device="cpu", local_files_only=True)
        print("[vendor.py] Hardware Acceleration: Running on CPU.", flush=True)

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
    Byte offsets are relative to the start of the string encoded as UTF-8.
    """
    sentences = nltk.sent_tokenize(text)
    
    # Further refine chunks: if NLTK returns a massive chunk (due to missing periods in PDFs),
    # force split it by newlines so we get granular sentence/line embeddings.
    refined_chunks = []
    for s in sentences:
        if len(s) > 150 and '\n' in s:
            for line in s.split('\n'):
                if line.strip():
                    refined_chunks.append(line.strip())
        else:
            if s.strip():
                refined_chunks.append(s.strip())
                
    results = []
    current_char_idx = 0
    for s_stripped in refined_chunks:
        start_char_idx = text.find(s_stripped, current_char_idx)
        if start_char_idx == -1:
            start_char_idx = current_char_idx
        current_char_idx = start_char_idx + len(s_stripped)
        byte_offset = len(text[:start_char_idx].encode('utf-8'))
        byte_length = len(s_stripped.encode('utf-8'))
        results.append((s_stripped, byte_offset, byte_length))
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
