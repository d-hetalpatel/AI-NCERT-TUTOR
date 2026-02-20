import streamlit as st
import os
import zipfile
import numpy as np
import faiss
import torch
import requests
import gdown
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================================
# CONFIG
# ==========================================================
FILE_ID = "1zrJOzLjnOIBuVVbTW0FsX38V6xIlpjV2"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"
TOP_K = 4
BATCH_SIZE = 64
OCR_DPI = 200  # higher = better quality but slower

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="NCERT AI Tutor", layout="wide")
st.title("📘 NCERT AI Tutor")
st.caption("Ask questions from NCERT textbooks using Retrieval-Augmented Generation")

# ==========================================================
# RECURSIVE ZIP EXTRACTOR
# ==========================================================
def extract_all_zips(folder):
    """Recursively extract any ZIP files found inside a folder."""
    found_new = True
    while found_new:
        found_new = False
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".zip"):
                    zip_path = os.path.join(root, file)
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
                    session = requests.Session()
                    URL = "https://drive.google.com/uc?export=download"
                    response = session.get(URL, params={"id": file_id}, stream=True)

                    token = next(
                        (v for k, v in response.cookies.items() if k.startswith("download_warning")),
                        None
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
                    st.error(f"Both download methods failed.\n\nError: {e2}")
                    st.stop()

    if not zipfile.is_zipfile(ZIP_PATH):
        os.remove(ZIP_PATH)
        st.error("Downloaded file is not a valid ZIP. Please download manually and place as `ncert.zip`.")
        st.stop()

    if not os.path.exists(EXTRACT_DIR):
        with st.spinner("Extracting outer ZIP..."):
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)

    with st.spinner("Extracting inner subject ZIPs..."):
        extract_all_zips(EXTRACT_DIR)

    pdf_files = []
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    st.info(f"📄 PDFs found: {len(pdf_files)}")
    return EXTRACT_DIR, pdf_files


data_path, pdf_files = download_and_extract(FILE_ID)

# ==========================================================
# OCR TEXT EXTRACTION
# ==========================================================
def extract_text_from_pdf(pdf_path):
    """Try direct text extraction first, fall back to OCR if needed."""
    text = ""

    # Attempt 1: direct text layer via PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            t = page.get_text()
            if t:
                text += t + "\n"
        doc.close()
    except Exception:
        pass

    if len(text.strip()) > 100:
        return text

    # Attempt 2: OCR via pytesseract
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            page_text = pytesseract.image_to_string(img, lang="eng")
            text += page_text + "\n"
        doc.close()
    except Exception:
        pass

    return text


@st.cache_resource
def load_documents_with_ocr(pdf_file_list):
    docs = []
    total = len(pdf_file_list)
    progress = st.progress(0, text="Extracting text from PDFs via OCR...")

    for i, path in enumerate(pdf_file_list):
        file = os.path.basename(path)
        try:
            text = extract_text_from_pdf(path)
            if text.strip():
                docs.append({"doc_id": file, "text": text})
        except Exception as e:
            st.warning(f"Failed on {file}: {e}")

        progress.progress((i + 1) / total, text=f"OCR: {i+1}/{total} — {file}")

    progress.empty()
    return docs


st.info("🔍 Extracting text via OCR (this takes several minutes on first run)...")
documents = load_documents_with_ocr(pdf_files)

if len(documents) == 0:
    st.error(
        "No text could be extracted even with OCR. "
        "Tesseract may not be installed on this server. "
        "Add a `packages.txt` file to your repo with the line: tesseract-ocr"
    )
    st.stop()

st.success(f"✅ Loaded {len(documents)} PDF files with text")

# ==========================================================
# CHUNKING
# ==========================================================
@st.cache_resource
def split_documents(docs, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "text": piece
            })
    return chunks


all_chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

if len(all_chunks) == 0:
    st.error("No text chunks were created.")
    st.stop()

st.success(f"✅ Created {len(all_chunks)} text chunks")

# ==========================================================
# BUILD VECTOR INDEX
# ==========================================================
@st.cache_resource(show_spinner=False)
def build_index(chunks, model_name):
    embed_model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]

    all_embeddings = []
    total = len(texts)
    progress = st.progress(0, text="Building vector index...")

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)
        progress.progress(
            min((i + BATCH_SIZE) / total, 1.0),
            text=f"Embedding chunks... {min(i + BATCH_SIZE, total)}/{total}"
        )

    progress.empty()
    embeddings = np.vstack(all_embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return embed_model, index, chunks


st.info("⚙️ Building vector index...")
embed_model, index, metadata = build_index(all_chunks, EMBED_MODEL_NAME)
st.success("✅ Vector index ready")

# ==========================================================
# LOAD GENERATION MODEL
# ==========================================================
@st.cache_resource
def load_generator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


tokenizer, gen_model = load_generator(GEN_MODEL_NAME)
st.success("✅ Generation model loaded")

# ==========================================================
# RETRIEVAL
# ==========================================================
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query]).astype("float32")
    D, I = index.search(q_emb, top_k)
    return [metadata[i] for i in I[0]]

# ==========================================================
# PROMPT BUILDER
# ==========================================================
def build_prompt(context_chunks, question):
    context = "\n\n".join([c["text"] for c in context_chunks])
    return (
        f"You are an AI tutor specializing in NCERT textbooks.\n"
        f"Answer ONLY using the provided context.\n"
        f"Be clear and concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )

# ==========================================================
# GENERATION
# ==========================================================
def generate_answer(query):
    retrieved = retrieve(query)
    if not retrieved:
        return "No relevant information found.", []

    prompt = build_prompt(retrieved, query)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=256)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip(), retrieved

# ==========================================================
# USER INTERFACE
# ==========================================================
st.markdown("---")
query = st.text_input("💬 Ask your question from NCERT:", placeholder="e.g. What is photosynthesis?")

if query:
    with st.spinner("Generating answer..."):
        answer, retrieved_chunks = generate_answer(query)

    st.markdown("## 📖 Answer")
    st.write(answer)

    st.markdown("## 📚 Top Retrieved Chunks Used to Generate the Answer")
    for i, chunk in enumerate(retrieved_chunks):
        with st.expander(f"Chunk {i + 1} — {chunk['doc_id']}"):
            st.write(chunk["text"])
