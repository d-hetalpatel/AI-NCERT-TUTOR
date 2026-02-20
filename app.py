import streamlit as st
import os
import zipfile
import numpy as np
import faiss
import torch
import requests
import gdown
from pypdf import PdfReader
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
                    # Extract into a subfolder named after the zip (without extension)
                    extract_to = os.path.join(root, os.path.splitext(file)[0])
                    if not os.path.exists(extract_to):
                        try:
                            with zipfile.ZipFile(zip_path, "r") as zf:
                                zf.extractall(extract_to)
                            found_new = True  # keep looping in case of deeper nesting
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

            # Method 1: gdown with fuzzy=True
            try:
                gdown.download(
                    id=file_id,
                    output=ZIP_PATH,
                    quiet=False,
                    fuzzy=True
                )
                success = os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 10000
            except Exception as e:
                st.warning(f"Primary download failed: {e}. Trying fallback...")

            # Method 2: requests session with confirm token
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
                        response = session.get(
                            URL, params={"id": file_id, "confirm": token}, stream=True
                        )
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
                        raise ValueError("Downloaded file too small — likely an HTML error page.")

                    st.success("Downloaded via fallback method.")

                except Exception as e2:
                    st.error(
                        f"Both download methods failed.\n\nError: {e2}\n\n"
                        f"**Manual fix:** Download the file from "
                        f"https://drive.google.com/file/d/{file_id}/view "
                        f"and place it as `{ZIP_PATH}` next to app.py, then rerun."
                    )
                    st.stop()

    # Validate ZIP
    if not zipfile.is_zipfile(ZIP_PATH):
        os.remove(ZIP_PATH)
        st.error(
            "The downloaded file is not a valid ZIP. "
            "Google likely blocked the download. "
            "Please manually download and place it as `ncert.zip` next to app.py."
        )
        st.stop()

    # Extract outer ZIP
    if not os.path.exists(EXTRACT_DIR):
        with st.spinner("Extracting outer ZIP..."):
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)

    # Recursively extract all inner ZIPs (subject ZIPs containing the actual PDFs)
    with st.spinner("Extracting inner subject ZIPs..."):
        extract_all_zips(EXTRACT_DIR)

    # Count PDFs after full extraction
    all_files = []
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            all_files.append(os.path.join(root, f))

    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
    st.info(f"📂 Total files found: {len(all_files)} | 📄 PDFs found: {len(pdf_files)}")

    if not pdf_files:
        st.error("Still no PDFs found after full extraction. Showing file tree:")
        st.code("\n".join(all_files[:60]))
        st.stop()

    return EXTRACT_DIR


data_path = download_and_extract(FILE_ID)

# ==========================================================
# LOAD DOCUMENTS
# ==========================================================
@st.cache_resource
def load_documents(folder):
    docs = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                try:
                    reader = PdfReader(path)
                    text = ""
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"

                    if text.strip():
                        docs.append({
                            "doc_id": file,
                            "text": text
                        })
                except Exception as e:
                    st.warning(f"Could not read {file}: {e}")
                    continue

    return docs


documents = load_documents(data_path)

if len(documents) == 0:
    st.error("No text could be extracted from the PDFs. They may be scanned image-based PDFs.")
    st.stop()

st.success(f"✅ Loaded {len(documents)} PDF files")

# ==========================================================
# CHUNKING
# ==========================================================
@st.cache_resource
def split_documents(docs, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

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
# BUILD VECTOR INDEX (BATCHED)
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
        emb = embed_model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )
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


st.info("⚙️ Building vector index (first run may take a few minutes)...")
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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=256
        )

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
