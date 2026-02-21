import streamlit as st
import os
import re
import zipfile
import pickle
import hashlib
import json
import numpy as np
import faiss
import torch
import requests
import gdown
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================================
# CONFIG
# ==========================================================
FILE_ID       = "1zrJOzLjnOIBuVVbTW0FsX38V6xIlpjV2"
ZIP_PATH      = "ncert.zip"
EXTRACT_DIR   = "ncert"

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 100          # FIX: skip garbage chunks shorter than this after cleaning

EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL_NAME    = "google/flan-t5-base"

RETRIEVE_K    = 20
RERANK_TOP_K  = 4            # reduced from 6 so chunks fit fully in prompt
BATCH_SIZE    = 64

INDEX_PATH    = "faiss_index.bin"
META_PATH     = "chunks_meta.pkl"
HASH_PATH     = "index_hash.txt"

# ==========================================================
# SUBJECT / CLASS CONSTANTS
# ==========================================================
SUBJECT_LIST = ["Auto", "Business Studies", "Economics", "Polity", "Psychology", "Sociology"]
CLASS_LIST   = ["Auto", "Class 11", "Class 12"]

FOLDER_SUBJECT_MAP = {
    "business studies": "Business Studies",
    "economics":        "Economics",
    "polity":           "Polity",
    "psychology":       "Psychology",
    "sociology":        "Sociology",
}

FILENAME_CODE_MAP = {
    "bs": "Business Studies",
    "ec": "Economics",
    "po": "Polity",
    "ps": "Psychology",
    "so": "Sociology",
}

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="NCERT AI Tutor", layout="wide")
st.title("📘 NCERT AI Tutor")
st.caption("Class 11 & 12 — Business Studies · Economics · Polity · Psychology · Sociology")

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.header("📚 Filter Your Search")
    st.caption("Leave as 'Auto' to search all books.")
    selected_class   = st.selectbox("Class",   CLASS_LIST,   index=0)
    selected_subject = st.selectbox("Subject", SUBJECT_LIST, index=0)
    selected_types   = st.multiselect(
        "Content Type",
        ["theory", "definition", "example", "exercise", "summary", "activity"],
        default=["theory", "definition", "example"],
        help="Filter chunks by content type detected inside the text."
    )
    st.divider()
    st.header("⚙️ Index")
    if st.button("🔄 Force Rebuild Index"):
        for p in [INDEX_PATH, META_PATH, HASH_PATH]:
            if os.path.exists(p):
                os.remove(p)
        st.cache_resource.clear()
        st.success("Cache cleared — refresh to rebuild.")
    st.divider()
    st.header("ℹ️ Model Info")
    st.caption(f"Embed: `{EMBED_MODEL_NAME}`")
    st.caption(f"Reranker: `{RERANK_MODEL_NAME}`")
    st.caption(f"Generator: `{GEN_MODEL_NAME}`")
    st.metric("FAISS candidates", RETRIEVE_K)
    st.metric("After reranking",  RERANK_TOP_K)

# ==========================================================
# RECURSIVE ZIP EXTRACTOR
# ==========================================================
def extract_all_zips(folder):
    """Recursively extract any nested ZIP files (subject folders)."""
    found_new = True
    while found_new:
        found_new = False
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".zip"):
                    zip_path   = os.path.join(root, file)
                    extract_to = os.path.join(root, os.path.splitext(file)[0])
                    if not os.path.exists(extract_to):
                        try:
                            with zipfile.ZipFile(zip_path, "r") as zf:
                                zf.extractall(extract_to)
                            found_new = True
                        except Exception as e:
                            st.warning(f"Could not extract {file}: {e}")

# ==========================================================
# DOWNLOAD + EXTRACT
# ==========================================================
@st.cache_resource
def download_and_extract(file_id):
    if not os.path.exists(ZIP_PATH):
        with st.spinner("Downloading NCERT dataset..."):
            success = False
            try:
                gdown.download(id=file_id, output=ZIP_PATH, quiet=False, fuzzy=True)
                success = os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 10000
            except Exception as e:
                st.warning(f"Primary download failed: {e}. Trying fallback...")

            if not success:
                try:
                    session  = requests.Session()
                    URL      = "https://drive.google.com/uc?export=download"
                    response = session.get(URL, params={"id": file_id}, stream=True)
                    token    = next(
                        (v for k, v in response.cookies.items()
                         if k.startswith("download_warning")), None
                    )
                    if token:
                        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
                    else:
                        response = session.get(
                            f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t",
                            stream=True
                        )
                    with open(ZIP_PATH, "wb") as f:
                        for chunk in response.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)
                    if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 10000:
                        raise ValueError("Downloaded file too small.")
                except Exception as e2:
                    st.error(f"Both download methods failed: {e2}")
                    st.stop()

    if not zipfile.is_zipfile(ZIP_PATH):
        os.remove(ZIP_PATH)
        st.error("Downloaded file is not a valid ZIP.")
        st.stop()

    if not os.path.exists(EXTRACT_DIR):
        with st.spinner("Extracting outer ZIP..."):
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(EXTRACT_DIR)

    with st.spinner("Extracting subject ZIPs..."):
        extract_all_zips(EXTRACT_DIR)

    pdf_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(EXTRACT_DIR)
        for f in files if f.lower().endswith(".pdf")
    ]
    st.info(f"📄 PDFs found: {len(pdf_files)}")
    return EXTRACT_DIR, pdf_files


data_path, pdf_files = download_and_extract(FILE_ID)

# ==========================================================
# METADATA HELPERS
# ==========================================================
def parse_metadata_from_folder(folder_path):
    """
    Primary source — reads folder name directly.
    'class 11 business studies' → Class 11, Business Studies
    """
    parts       = folder_path.lower().replace("\\", "/").split("/")
    class_level = "Unknown"
    subject     = "Unknown"

    for part in parts:
        m = re.search(r'class\s*(1[12])', part)
        if m:
            class_level = f"Class {m.group(1)}"
        for keyword, canonical in FOLDER_SUBJECT_MAP.items():
            if keyword in part:
                subject = canonical
                break

    return class_level, subject


def parse_metadata_from_filename(filename):
    """
    Fallback — decode NCERT 2-letter subject code from filename.

    FIX 1: pattern now requires digit after subject code to avoid false matches
            e.g. leps104.pdf → 'ps' followed by '1' → Psychology (correct)
            without fix: leps matched 'ps' but also caught wrong codes

    FIX 2: chapter numbers > 20 are catalog/ISBN codes, not real chapters
            e.g. leps104 → 104 is discarded, not stored as 'Chapter 104'
    """
    name    = filename.lower().replace(".pdf", "")
    subject = "Unknown"
    chapter = "Unknown"

    # Must be followed by a digit to avoid false positives
    m = re.match(r'^[lik]e([a-z]{2})\d', name)
    if m:
        subject = FILENAME_CODE_MAP.get(m.group(1), "Unknown")

    # Only keep plausible chapter numbers (1-20), discard catalog codes
    numbers       = re.findall(r'\d+', name)
    real_chapters = [n for n in numbers if int(n) <= 20]
    if real_chapters:
        chapter = f"Chapter {real_chapters[0]}"

    return subject, chapter


def get_full_metadata(full_path):
    folder   = os.path.dirname(full_path)
    filename = os.path.basename(full_path)

    # Folder name is ground truth for class + subject
    class_level, subject = parse_metadata_from_folder(folder)

    # Only use filename as fallback if folder gave nothing
    if subject == "Unknown":
        subject, _ = parse_metadata_from_filename(filename)

    _, chapter = parse_metadata_from_filename(filename)

    return {
        "filename":    filename,
        "subject":     subject,
        "class_level": class_level,
        "chapter":     chapter,
    }

# ==========================================================
# SECTION TYPE DETECTION
# ==========================================================
def detect_section_type(text):
    t = text.lower()
    if any(k in t for k in ["exercise", "very short answer", "answer the following",
                             "fill in the blank", "true or false", "questions for practice"]):
        return "exercise"
    if any(k in t for k in ["is defined as", "is called", "refers to",
                             "may be defined", "can be defined", "definition"]):
        return "definition"
    if any(k in t for k in ["for example", "for instance", "e.g.",
                             "case study", "let us understand", "illustration"]):
        return "example"
    if any(k in t for k in ["summary", "key points", "in this chapter",
                             "we have learnt", "let us recapitulate"]):
        return "summary"
    if any(k in t for k in ["activity", "project", "do it yourself",
                             "think and discuss", "intext question"]):
        return "activity"
    return "theory"

# ==========================================================
# CLEAN TEXT
# ==========================================================
def clean_chunk_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n([a-z])', r' \1', text)
    text = re.sub(r'\n?\d+\s*\n', '\n', text)
    text = re.sub(r'Reprint \d{4}-\d{2}', '', text)
    text = re.sub(r'Prelims\.indd\s*\d+.*', '', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()

# ==========================================================
# ENRICH CHUNK TEXT WITH METADATA HEADER
# ==========================================================
def enrich_chunk_text(chunk):
    prefix = (
        f"[{chunk['class_level']} | {chunk['subject']} | "
        f"{chunk['chapter']} | {chunk['section_type'].upper()}]\n\n"
    )
    return prefix + clean_chunk_text(chunk["text"])

# ==========================================================
# LOAD DOCUMENTS via PyMuPDF
# ==========================================================
@st.cache_resource
def load_documents(pdf_file_list):
    docs  = []
    total = len(pdf_file_list)
    prog  = st.progress(0, text="Reading PDFs...")

    for i, path in enumerate(pdf_file_list):
        file = os.path.basename(path)
        try:
            doc  = fitz.open(path)
            text = "".join(
                page.get_text() + "\n"
                for page in doc
                if page.get_text()
            )
            doc.close()
            if text.strip():
                docs.append({
                    "doc_id":    file,
                    "full_path": path,
                    "text":      text
                })
        except Exception as e:
            st.warning(f"Could not read {file}: {e}")

        prog.progress((i + 1) / total, text=f"Reading {i+1}/{total} — {file}")

    prog.empty()
    return docs


st.info("📖 Reading PDFs...")
documents = load_documents(pdf_files)

if not documents:
    st.error("No text could be extracted from the PDFs.")
    st.stop()

st.success(f"✅ Loaded {len(documents)} PDF files")

# ==========================================================
# STABLE HASH (cache invalidation)
# ==========================================================
def get_docs_hash(docs):
    fp = [{"doc_id": d["doc_id"], "len": len(d["text"])}
          for d in sorted(docs, key=lambda x: x["doc_id"])]
    return hashlib.md5(json.dumps(fp, sort_keys=True).encode()).hexdigest()


docs_hash = get_docs_hash(documents)

# ==========================================================
# CHUNKING WITH FULL METADATA + GARBAGE FILTER
# ==========================================================
def split_documents(docs, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = []
    for doc in docs:
        meta   = get_full_metadata(doc["full_path"])
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):

            # FIX: clean first, then check length
            # Skips blank pages, header-only pages, numbered list fragments
            # that were showing up as empty source expanders
            cleaned = clean_chunk_text(piece)
            if len(cleaned) < MIN_CHUNK_LEN:
                continue

            chunks.append({
                "doc_id":       doc["doc_id"],
                "chunk_id":     f"{doc['doc_id']}_chunk_{i}",
                "text":         piece,
                "subject":      meta["subject"],
                "class_level":  meta["class_level"],
                "chapter":      meta["chapter"],
                "section_type": detect_section_type(piece),
            })
    return chunks


@st.cache_resource
def split_documents_cached(docs_hash, chunk_size, overlap):
    # docs_hash in signature → cache busts automatically when PDFs change
    return split_documents(documents, chunk_size, overlap)


all_chunks = split_documents_cached(docs_hash, CHUNK_SIZE, CHUNK_OVERLAP)

if not all_chunks:
    st.error("No text chunks were created.")
    st.stop()

st.success(f"✅ Created {len(all_chunks)} text chunks")

# Sanity check — show detected subjects/classes
det_subjects = sorted(set(c["subject"]     for c in all_chunks if c["subject"]     != "Unknown"))
det_classes  = sorted(set(c["class_level"] for c in all_chunks if c["class_level"] != "Unknown"))
if det_subjects:
    st.info(f"📖 Detected → {', '.join(det_classes)} | {', '.join(det_subjects)}")

# ==========================================================
# FAISS DISK PERSISTENCE HELPERS
# ==========================================================
def index_is_valid(h):
    if not all(os.path.exists(p) for p in [INDEX_PATH, META_PATH, HASH_PATH]):
        return False
    with open(HASH_PATH) as f:
        return f.read().strip() == h


def save_index(idx, chunks, h):
    faiss.write_index(idx, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)
    with open(HASH_PATH, "w") as f:
        f.write(h)


def load_index_safe(h):
    try:
        if index_is_valid(h):
            idx = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                chunks = pickle.load(f)
            assert idx.ntotal > 0 and len(chunks) > 0
            return idx, chunks
    except Exception as e:
        st.warning(f"Saved index invalid ({e}), rebuilding...")
        for p in [INDEX_PATH, META_PATH, HASH_PATH]:
            if os.path.exists(p):
                os.remove(p)
    return None, None

# ==========================================================
# BUILD VECTOR INDEX  (cosine via IndexFlatIP + L2 norm)
# ==========================================================
@st.cache_resource(show_spinner=False)
def build_index(docs_hash, model_name):
    embed_model = SentenceTransformer(model_name)

    idx, chunks = load_index_safe(docs_hash)
    if idx is not None:
        st.success("⚡ Index loaded from disk instantly!")
        return embed_model, idx, chunks

    st.info("Building vector index (first run only — saved for future use)...")
    texts    = [c["text"] for c in all_chunks]
    emb_list = []
    total    = len(texts)
    prog     = st.progress(0, text="Building vector index...")

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb   = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        emb_list.append(emb)
        prog.progress(
            min((i + BATCH_SIZE) / total, 1.0),
            text=f"Embedding... {min(i + BATCH_SIZE, total)}/{total}"
        )

    prog.empty()

    embeddings = np.vstack(emb_list).astype("float32")
    faiss.normalize_L2(embeddings)

    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)

    save_index(idx, all_chunks, docs_hash)
    st.success("✅ Index built and saved to disk!")
    return embed_model, idx, all_chunks


st.info("⚙️ Loading vector index...")
embed_model, faiss_index, metadata = build_index(docs_hash, EMBED_MODEL_NAME)

with st.sidebar:
    if os.path.exists(INDEX_PATH):
        st.caption(f"📦 Index: {os.path.getsize(INDEX_PATH)/1e6:.1f} MB")
        st.caption(f"📄 Chunks: {len(all_chunks)}")

# ==========================================================
# CROSS-ENCODER RERANKER
# ==========================================================
@st.cache_resource
def load_reranker(model_name):
    return CrossEncoder(model_name, max_length=512)


reranker = load_reranker(RERANK_MODEL_NAME)
st.success("✅ Cross-encoder reranker loaded")

# ==========================================================
# GENERATION MODEL (flan-t5-base)
# ==========================================================
@st.cache_resource
def load_generator(model_name):
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tok, model


tokenizer, gen_model = load_generator(GEN_MODEL_NAME)
st.success("✅ Generation model loaded")

# ==========================================================
# AUTO-DETECT SUBJECT/CLASS FROM QUESTION
# ==========================================================
def auto_detect_from_question(question):
    q  = question.lower()
    kw = {
        "Business Studies": [
            "management", "planning", "organising", "staffing", "directing",
            "controlling", "entrepreneur", "marketing", "finance", "consumer",
            "stock exchange", "recruitment", "motivation", "leadership",
            "communication", "supervision", "business environment"
        ],
        "Economics": [
            "gdp", "demand", "supply", "market", "price", "inflation", "deficit",
            "budget", "fiscal", "monetary", "national income", "poverty",
            "unemployment", "elasticity", "consumer equilibrium", "producer",
            "revenue", "cost", "profit", "macro", "micro"
        ],
        "Polity": [
            "constitution", "parliament", "president", "prime minister",
            "fundamental rights", "directive principles", "judiciary",
            "legislature", "executive", "election", "federalism", "preamble",
            "lok sabha", "rajya sabha", "governor", "supreme court", "citizenship"
        ],
        "Psychology": [
            "behaviour", "cognition", "perception", "memory", "learning",
            "intelligence", "personality", "attitude", "motivation", "emotion",
            "stress", "therapy", "disorder", "consciousness", "sensation"
        ],
        "Sociology": [
            "society", "culture", "caste", "class", "gender", "tribe",
            "institution", "socialisation", "community", "rural", "urban",
            "inequality", "stratification", "kinship", "family", "religion",
            "social change"
        ],
    }

    detected_subject = "Unknown"
    for subj, words in kw.items():
        if any(w in q for w in words):
            detected_subject = subj
            break

    detected_class = "Unknown"
    if any(k in q for k in ["class 11", "11th", "first year"]):
        detected_class = "Class 11"
    elif any(k in q for k in ["class 12", "12th", "second year", "board"]):
        detected_class = "Class 12"

    return detected_subject, detected_class

# ==========================================================
# HyDE — Hypothetical Document Embedding
# Generate a fake answer with flan-t5, then embed THAT for retrieval.
# Bridges the gap between question-space and passage-space embeddings.
# ==========================================================
def hypothetical_answer(question):
    prompt = f"Write a short explanation about: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_new_tokens=80, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ==========================================================
# DEDUP — Remove near-duplicate chunks caused by chunk overlap
# Prevents the same passage appearing multiple times in retrieved results.
# ==========================================================
def deduplicate_chunks(chunks, threshold=0.8):
    seen, result = [], []
    for chunk in chunks:
        text  = clean_chunk_text(chunk["text"])
        words = set(text.split())
        is_dup = any(
            len(words & set(s.split())) / max(len(words), 1) > threshold
            for s in seen
        )
        if not is_dup:
            seen.append(text)
            result.append(chunk)
    return result

# ==========================================================
# RETRIEVAL → FILTER → DEDUP → RERANK
# ==========================================================
def retrieve_and_rerank(query, sel_subject, sel_class, sel_types):
    # HyDE: embed a hypothetical answer instead of the raw question
    # for better semantic alignment with passage embeddings
    hyp   = hypothetical_answer(query)
    q_emb = embed_model.encode([hyp], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    _, I       = faiss_index.search(q_emb, RETRIEVE_K)
    candidates = [metadata[i] for i in I[0] if i < len(metadata)]

    if not candidates:
        return []

    filtered = []
    for c in candidates:
        if sel_subject != "Auto" and c["subject"]     != sel_subject: continue
        if sel_class   != "Auto" and c["class_level"] != sel_class:   continue
        if sel_types   and c["section_type"] not in sel_types:        continue
        filtered.append(c)

    if not filtered:
        st.warning("⚠️ Filters too strict — showing unfiltered results.")
        filtered = candidates

    # Dedup before reranking — removes overlap-artifact near-duplicates
    filtered = deduplicate_chunks(filtered)

    pairs  = [[query, clean_chunk_text(c["text"])] for c in filtered]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, filtered), key=lambda x: x[0], reverse=True)

    return [c for _, c in ranked[:RERANK_TOP_K]]

# ==========================================================
# GROUNDING METADATA FOR PROMPT
# ==========================================================
def extract_context_metadata(chunks, question, sel_subject, sel_class):
    if sel_subject != "Auto" and sel_class != "Auto":
        return sel_subject, sel_class

    subjects = set(c["subject"]     for c in chunks if c["subject"]     != "Unknown")
    classes  = set(c["class_level"] for c in chunks if c["class_level"] != "Unknown")

    subj_str = ", ".join(subjects) if subjects else ""
    cls_str  = ", ".join(classes)  if classes  else ""

    if not subj_str or not cls_str:
        ds, dc   = auto_detect_from_question(question)
        subj_str = subj_str or ds
        cls_str  = cls_str  or dc

    return subj_str or "NCERT", cls_str or "Class 11/12"

# ==========================================================
# PROMPT BUILDER
# flan-t5 works best with short, direct prompts — keep context tight
# ==========================================================
def build_prompt(context_chunks, question, sel_subject, sel_class):
    subj, cls = extract_context_metadata(context_chunks, question, sel_subject, sel_class)

    # Use only cleaned text (no metadata prefix) to save tokens for flan-t5
    context = "\n\n".join(
        f"Passage {i+1}: {clean_chunk_text(c['text'])}"
        for i, c in enumerate(context_chunks)
    )

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above. "
        f"Quote relevant parts of the passages in your answer from best chunk.\n"
        f"Answer:"
    )

# ==========================================================
# POST-PROCESS ANSWER
# ==========================================================
def clean_answer(text):
    sentences = text.split('. ')
    seen, out = set(), []
    for s in sentences:
        if s.strip() not in seen:
            seen.add(s.strip())
            out.append(s.strip())
    result = '. '.join(out)
    if result and result[-1] not in '.!?':
        parts = result.rsplit('.', 1)
        if len(parts) > 1:
            result = parts[0] + '.'
    return result.strip()

# ==========================================================
# HIGHLIGHT QUERY TERMS IN SOURCE TEXT
# ==========================================================
def highlight_terms(text, query):
    for word in query.split():
        if len(word) > 3:
            text = re.sub(f"(?i)({re.escape(word)})", r"**\1**", text)
    return text

# ==========================================================
# FULL PIPELINE
# ==========================================================
def generate_answer(query, sel_subject, sel_class, sel_types):
    retrieved = retrieve_and_rerank(query, sel_subject, sel_class, sel_types)

    if not retrieved:
        return "No relevant information found.", []

    prompt = build_prompt(retrieved, query, sel_subject, sel_class)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=50,        # force at least 50 tokens — avoids fragment answers
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0        # penalise short answers
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_answer(raw), retrieved

# ==========================================================
# UI
# ==========================================================
st.markdown("---")
query = st.text_input(
    "💬 Ask your question:",
    placeholder="e.g. What are the features of planning? / Explain price elasticity of demand"
)

if query:
    with st.spinner("Retrieving, reranking and generating answer..."):
        answer, ret_chunks = generate_answer(
            query, selected_subject, selected_class, selected_types
        )

    if selected_subject == "Auto" or selected_class == "Auto":
        subj_used, cls_used = extract_context_metadata(
            ret_chunks, query, selected_subject, selected_class
        )
        st.info(
            f"🔍 Auto-detected: **{subj_used}** | **{cls_used}**  \n"
            f"*Use sidebar filters to override if wrong.*"
        )

    st.markdown("## 📖 Answer")
    if "don't know" in answer.lower():
        st.warning(answer)
    else:
        st.success(answer)

    st.markdown(f"## 📚 Top {RERANK_TOP_K} Sources (After Reranking)")
    for i, chunk in enumerate(ret_chunks):
        label = (
            f"Source {i+1}  —  "
            f"{chunk.get('class_level','?')}  |  "
            f"{chunk.get('subject','?')}  |  "
            f"{chunk.get('chapter','?')}  |  "
            f"{chunk.get('section_type','?').upper()}"
        )
        with st.expander(label):
            st.caption(f"📄 `{chunk.get('doc_id','?')}`")
            # FIX: removed duplicate st.write — only render highlighted markdown once
            st.markdown(highlight_terms(clean_chunk_text(chunk["text"]), query))
