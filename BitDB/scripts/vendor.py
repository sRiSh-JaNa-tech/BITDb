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
# Option 2: Tokenizer-Native Windowing & Cross-Page Buffering
# ─────────────────────────────────────────────────────────────────────

_carry_over_buffer = ""
_current_doc_pages = 0
_current_page_idx = 0
_fast_tokenizer = None

def reset_document_chunker(total_pages: int = 0):
    global _carry_over_buffer, _current_doc_pages, _current_page_idx
    _carry_over_buffer = ""
    _current_doc_pages = total_pages
    _current_page_idx = 0

def _get_fast_tokenizer():
    global _fast_tokenizer
    if _fast_tokenizer is not None:
        return _fast_tokenizer

    # Priority 1: Direct Rust Tokenizer from tokenizer.json (sub-millisecond speed)
    tok_json_candidates = [
        os.path.join(_OV_MODEL_PATH, "tokenizer.json"),
        os.path.join(_MODEL_PATH, "tokenizer.json"),
    ]
    for p in tok_json_candidates:
        if os.path.isfile(p):
            try:
                from tokenizers import Tokenizer
                t = Tokenizer.from_file(p)
                t.no_truncation()
                _fast_tokenizer = t
                return _fast_tokenizer
            except Exception:
                pass

    # Priority 2: In-memory tokenizer from OpenVINO or SentenceTransformer
    if ov_tokenizer is not None:
        _fast_tokenizer = ov_tokenizer
        return _fast_tokenizer
    if st_model is not None and hasattr(st_model, "tokenizer"):
        _fast_tokenizer = st_model.tokenizer
        return _fast_tokenizer

    # Priority 3: Fallback to transformers AutoTokenizer
    from transformers import AutoTokenizer
    _fast_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME_OR_PATH, local_files_only=_HAS_LOCAL_MINILM)
    return _fast_tokenizer

def _chunk_text_to_offsets(text: str, window_tokens: int = 96, stride_tokens: int = 64) -> list:
    """
    Option 2: Tokenizer-Native Windowing (Rust compiled subword sliding window).
    Non-greedy, model-exact token partitions with cross-page carry-over.
    """
    global _carry_over_buffer, _current_page_idx, _current_doc_pages
    _current_page_idx += 1
    is_last_page = (_current_doc_pages > 0 and _current_page_idx >= _current_doc_pages)

    # Cross-page continuous buffering: prepend carry-over from previous page
    if _carry_over_buffer:
        full_text = _carry_over_buffer + " " + text
        _carry_over_buffer = ""
    else:
        full_text = text

    full_text = full_text.strip()
    if not full_text:
        return []

    tokenizer = _get_fast_tokenizer()

    # Fast path: Native Rust Tokenizer instance
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "no_truncation"):
        enc = tokenizer.encode(full_text)
        n_tokens = len(enc.ids)
        if n_tokens < 20:
            if not is_last_page:
                _carry_over_buffer = full_text
            elif len(full_text) >= 40:
                return [(full_text, 0, len(full_text.encode('utf-8')))]
            return []

        chunks = []
        i = 0
        while i < n_tokens:
            w_end = min(i + window_tokens, n_tokens)
            w_len = w_end - i

            # If remaining tail tokens at page end are too short, carry over to next page
            if w_len < 35 and not is_last_page and i > 0:
                v_offs = [(s, e) for s, e in enc.offsets[i:w_end] if not (s == 0 and e == 0)]
                if v_offs:
                    _carry_over_buffer = full_text[v_offs[0][0]:v_offs[-1][1]].strip()
                break

            v_offs = [(s, e) for s, e in enc.offsets[i:w_end] if not (s == 0 and e == 0)]
            if v_offs:
                chunk = full_text[v_offs[0][0]:v_offs[-1][1]].strip()
                if len(chunk) >= 50:
                    byte_len = len(chunk.encode('utf-8'))
                    chunks.append((chunk, 0, byte_len))
            if w_end >= n_tokens:
                break
            i += stride_tokens
        return chunks
    else:
        # Fallback for HuggingFace PreTrainedTokenizerFast
        out = tokenizer(
            full_text,
            max_length=window_tokens,
            stride=window_tokens - stride_tokens,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False
        )
        chunks = []
        for ids, offsets in zip(out.input_ids, out.offset_mapping):
            valid_offsets = [(s, e) for s, e in offsets if not (s == 0 and e == 0)]
            if valid_offsets:
                chunk = full_text[valid_offsets[0][0]:valid_offsets[-1][1]].strip()
                if len(chunk) >= 50:
                    chunks.append((chunk, 0, len(chunk.encode('utf-8'))))
        return chunks

def chunk_text_with_offsets(text: str) -> list:
    """
    Chunks raw text into Rust-tokenized subword windows with cross-page carry-over.
    """
    if not text or not text.strip():
        return []
    return _chunk_text_to_offsets(text)

# ─────────────────────────────────────────────────────────────────────
# PDF extraction — Prototype-3 addition
# ─────────────────────────────────────────────────────────────────────

def pdf_to_text_pages(filepath: str) -> list:
    """
    Extracts text from a PDF file page by page and initializes
    cross-page document buffering state.
    """
    try:
        from pdf_extractor import pdf_to_text_pages as _extract
        pages = _extract(filepath)
        reset_document_chunker(len(pages))
        return pages
    except ImportError as e:
        print(f"[vendor.py] ERROR: pdf_extractor not available: {e}")
        return []
    except Exception as e:
        print(f"[vendor.py] ERROR in pdf_to_text_pages: {e}")
        return []
