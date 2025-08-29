import sys
import streamlit as st
import numpy as np
import google.generativeai as genai
from config import GOOGLE_API_KEY  # load from .env via config.py

try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader


# ------------------------
# Models
# ------------------------
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "models/gemini-1.5-flash"

genai.configure(api_key=GOOGLE_API_KEY)


# ------------------------
# Helpers
# ------------------------
def read_pdf_file(file) -> list:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages


def chunk_text(pages, chunk_size: int = 1200, chunk_overlap: int = 200):
    chunks = []
    for page_num, txt in pages:
        if not txt:
            continue
        words = txt.split()
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append({"text": chunk_text, "meta": {"page": page_num}})
            if end == len(words):
                break
            start = max(0, end - chunk_overlap)
    return chunks


def embed_texts(texts):
    vectors = []
    for t in texts:
        resp = genai.embed_content(model=EMBED_MODEL, content=t)
        vectors.append(resp["embedding"])
    return np.array(vectors, dtype="float32")


def build_numpy_index(chunks):
    texts = [c["text"] for c in chunks]
    mat = embed_texts(texts)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    mat = mat / norms
    return {"mat": mat, "chunks": chunks}


def search(index_data, query: str, k: int = 5):
    mat = index_data["mat"]
    chunks = index_data["chunks"]
    q_vec = embed_texts([query]).astype("float32")[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    sims = mat @ q_vec
    topk_idx = np.argsort(-sims)[:k]
    results = []
    for i in topk_idx:
        sc = float(sims[i])
        ch = chunks[i]
        results.append({"score": sc, "text": ch["text"], "page": ch["meta"]["page"]})
    return results


def make_system_prompt() -> str:
    return (
        "You are a careful assistant that answers using ONLY the provided PDF context.\n"
        "Rules:\n"
        "- If the answer is not in the context, say: 'I couldn't find this in the document.'\n"
        "- Be concise and factual.\n"
        "- Cite page numbers in parentheses like (p. 12).\n"
    )


def build_context_snippet(hits) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[Page {h['page']}]\n{h['text']}")
    return "\n\n".join(blocks)


def answer_with_context(query: str, hits) -> str:
    context = build_context_snippet(hits)
    prompt = (
        f"{make_system_prompt()}\n\n"
        f"Context from the PDF (ranked):\n{context}\n\n"
        f"User question: {query}\n"
        f"Answer:"
    )
    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)
    return resp.text.strip() if hasattr(resp, "text") and resp.text else ""


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="PDF RAG with Gemini", page_icon="📄", layout="wide")

st.title("📄 PDF Chatbot (RAG + Gemini)")
st.caption("Upload a PDF, build an index, and chat with it. Keys are loaded securely from `.env`.")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    chunk_size = st.number_input("Chunk size (words)", min_value=200, max_value=2000, value=1200, step=100)
    overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=800, value=200, step=50)
    k = st.slider("Top-K passages", min_value=1, max_value=10, value=5)

# Initialize session state
if "index_data" not in st.session_state:
    st.session_state.index_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def build_index_action():
    if not uploaded_file:
        st.error("Please upload a PDF first.")
        return

    with st.spinner("📖 Reading and indexing PDF..."):
        pages = read_pdf_file(uploaded_file)
        if sum(len(p[1].strip()) for p in pages) == 0:
            st.error("No extractable text found in this PDF.")
            return
        chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=overlap)
        st.session_state.index_data = build_numpy_index(chunks)

    st.success(f"✅ Indexed {len(chunks)} chunks from {len(pages)} pages.")


if st.button("Build / Rebuild Index"):
    build_index_action()

# ------------------------
# Chat Interface
# ------------------------
chat_container = st.container()

with chat_container:
    st.markdown(
        """
        <style>
        .chat-box {
            max-height: 500px;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 0.75rem;
            background-color: #fafafa;
        }
        .user-msg {
            background-color: #DCF8C6;
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            margin-bottom: 0.5rem;
            text-align: right;
        }
        .bot-msg {
            background-color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            margin-bottom: 0.5rem;
            text-align: left;
            border: 1px solid #ddd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Fixed input at bottom
query = st.text_input("💬 Ask a question", key="chat_input")
if st.button("Send"):
    if not st.session_state.index_data:
        st.warning("Please build the index first.")
    elif query:
        st.session_state.messages.append({"role": "user", "content": query})
        hits = search(st.session_state.index_data, query, k=k)
        answer = answer_with_context(query, hits) or "(No answer)"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.experimental_rerun()
