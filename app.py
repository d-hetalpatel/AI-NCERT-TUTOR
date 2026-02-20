import streamlit as st
import os
import zipfile
import numpy as np
import faiss
import torch
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================================
# CONFIG
# ==========================================================
#FILE_ID = "1toFD-1u6BSpdDU-cop12nne2ysPgPHM0"
#ZIP_PATH = "ncert.zip"
#EXTRACT_DIR = "ncert_extracted"

FILE_ID = "1zrJOzLjnOIBuVVbTW0FsX38V6xIlpjV2"
ZIP_PATH = f"ncert_{FILE_ID}.zip"
EXTRACT_DIR = f"ncert_extracted_{FILE_ID}"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"  # Cloud safe
TOP_K = 4
BATCH_SIZE = 64

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="NCERT AI Tutor", layout="wide")
st.title("📘 NCERT AI Tutor")
st.caption("Ask questions from NCERT textbooks using Retrieval-Augmented Generation")

# ==========================================================
# DOWNLOAD + EXTRACT
# ==========================================================
@st.cache_resource
def download_and_extract(file_id):
    if not os.path.exists(ZIP_PATH):
        with st.spinner("Downloading NCERT dataset..."):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                ZIP_PATH,
                quiet=False
            )

    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

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
                except:
                    continue

    return docs


documents = load_documents(data_path)
st.success(f"Loaded {len(documents)} PDF files")

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
        for i, chunk in enumerate(pieces):
            chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "text": chunk
            })

    return chunks


all_chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
st.success(f"Created {len(all_chunks)} text chunks")

# ==========================================================
# BUILD VECTOR INDEX (BATCHED)
# ==========================================================
@st.cache_resource(show_spinner=False)
def build_index(chunks, model_name):

    embed_model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]

    all_embeddings = []
    total = len(texts)

    progress = st.progress(0)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        emb = embed_model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        all_embeddings.append(emb)
        progress.progress(min((i + BATCH_SIZE) / total, 1.0))

    progress.empty()

    embeddings = np.vstack(all_embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return embed_model, index, chunks


st.info("Building vector index (first run may take a few minutes)...")
embed_model, index, metadata = build_index(all_chunks, EMBED_MODEL_NAME)
st.success("Vector index ready")

# ==========================================================
# LOAD GENERATION MODEL
# ==========================================================
@st.cache_resource
def load_generator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


tokenizer, model = load_generator(GEN_MODEL_NAME)

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

    return f"""
You are an AI tutor specializing in NCERT textbooks.
Answer ONLY using the provided context.
Be clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""

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
        outputs = model.generate(
            **inputs,
            max_length=256
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.strip(), retrieved

# ==========================================================
# USER INTERFACE
# ==========================================================
query = st.text_input("Ask your question from NCERT:")

if query:
    with st.spinner("Generating answer..."):
        answer, retrieved_chunks = generate_answer(query)

    # Answer
    st.markdown("## 📖 Answer")
    st.write(answer)

    # Retrieved Context Section
    st.markdown("## 📚 Top K Retrieved Chunks Used to Generate the Answer")

    for i, chunk in enumerate(retrieved_chunks):
        with st.expander(f"Chunk {i+1} — {chunk['doc_id']}"):
            st.write(chunk["text"])
