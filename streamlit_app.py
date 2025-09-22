# streamlit_App.py
import streamlit as st
import numpy as np
import google.generativeai as genai
from config import GOOGLE_API_KEY

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
# Default Parameters
# ------------------------
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200
DEFAULT_TOP_K = 5

# ------------------------
# PDF Processing
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

def chunk_text(pages, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_OVERLAP):
    chunks = []
    for page_num, txt in pages:
        if not txt.strip():
            continue
        words = txt.split()
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_words = words[start:end]
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

def search(index_data, query, k=DEFAULT_TOP_K):
    mat = index_data["mat"]
    chunks = index_data["chunks"]
    q_vec = embed_texts([query])[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    sims = mat @ q_vec
    topk_idx = np.argsort(-sims)[:k]
    return [{"score": float(sims[i]), "text": chunks[i]["text"], "page": chunks[i]["meta"]["page"]} for i in topk_idx]

# ------------------------
# Prompt / Response Handling
# ------------------------
def make_system_prompt() -> str:
    return (
        "You are a precise assistant that answers using ONLY the PDF context provided.\n"
        "Rules:\n"
        "- Answer concisely but in a structured GPT-like format (headings, bullets, tables if needed).\n"
        "- Cite page numbers in parentheses like (p. 12).\n"
        "- If user asks to generate exam questions from the PDF, provide 5-10 well-formed questions with answers if possible.\n"
        "- If answer is not in the context, say: 'I couldn't find this in the document.'"
    )

def build_context_snippet(hits) -> str:
    return "\n\n".join([f"[Page {h['page']}]\n{h['text']}" for h in hits])

def answer_with_mem0(query, hits):
    pdf_context = build_context_snippet(hits)
    chat_history = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_history += f"{role}: {msg['content']}\n"

    prompt = (
        f"{make_system_prompt()}\n\n"
        f"PDF context (top {DEFAULT_TOP_K} chunks):\n{pdf_context}\n\n"
        f"Conversation so far:\n{chat_history}\n"
        f"User question: {query}\n"
        f"Assistant:"
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)
    return resp.text.strip() if hasattr(resp, "text") else "(No answer)"

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDF (RAG + mem0)")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "index_data" not in st.session_state:
    st.session_state.index_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file and st.session_state.index_data is None:
    with st.spinner("📖 Reading and indexing PDF..."):
        pages = read_pdf_file(uploaded_file)
        if sum(len(p[1].strip()) for p in pages) == 0:
            st.error("No extractable text found in this PDF.")
        else:
            chunks = chunk_text(pages)
            st.session_state.index_data = build_numpy_index(chunks)
            st.success(f"✅ Indexed {len(chunks)} chunks from {len(pages)} pages.")

# Chat panel styling
st.markdown("""
<style>
.chat-box { max-height: 600px; overflow-y: auto; padding: 1rem; border-radius: 1rem; background-color: #343541; }
.user-msg { background-color: #10a37f; color: white; padding: 0.7rem 1rem; border-radius: 1rem; margin-bottom: 0.5rem; text-align: right; }
.bot-msg { background-color: #444654; color: white; padding: 0.7rem 1rem; border-radius: 1rem; margin-bottom: 0.5rem; text-align: left; }
</style>
""", unsafe_allow_html=True)

# Chat display
chat_container = st.container()
with chat_container:
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Chat input
if query := st.chat_input("💬 Ask something about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": query})
    if not st.session_state.index_data:
        answer = "⚠️ Please upload a PDF first."
    else:
        hits = search(st.session_state.index_data, query, k=DEFAULT_TOP_K)
        answer = answer_with_mem0(query, hits)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
