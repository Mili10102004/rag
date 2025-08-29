# app.py
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import numpy as np
from config import GOOGLE_API_KEY

# ---------------------------
# Configure Gemini
# ---------------------------
genai.configure(api_key=GOOGLE_API_KEY)

# Embedding + Model
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"

# ---------------------------
# Helper functions
# ---------------------------

def extract_text_from_pdf(uploaded_files):
    """Extract text from multiple uploaded PDFs"""
    text = ""
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def create_chunks(text, chunk_size=500):
    """Split text into smaller chunks"""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def embed_text(text):
    """Convert single text into embedding"""
    embeddings = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return np.array(embeddings["embedding"])


def retrieve(query, stored_chunks, stored_embeddings, top_k=3):
    """Retrieve most relevant chunks for a query"""
    query_embedding = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    similarities = np.dot(stored_embeddings, query_embedding) / (
        np.linalg.norm(stored_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = similarities.argsort()[-top_k:][::-1]
    return [stored_chunks[i] for i in top_indices]


def generate_answer(system_prompt, user_query, context_chunks):
    """Generate answer using Gemini with system + user prompt"""
    prompt = f"""
    System: {system_prompt}

    Context from PDFs:
    {' '.join(context_chunks)}

    User: {user_query}
    Assistant:
    """
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)
    return response.text


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="RAG PDF Assistant", layout="wide")

st.title("📄 RAG-based Multi-PDF Assistant")

# Session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant. Answer queries truthfully using only the PDF context."

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        text = extract_text_from_pdf(uploaded_files)
        st.session_state.chunks = create_chunks(text)
        st.session_state.embeddings = np.array(
            [embed_text(chunk) for chunk in st.session_state.chunks]
        )
    st.success("✅ PDFs processed successfully!")


# Chat UI
st.subheader("💬 Chat with your PDFs")



# Show chat messages
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")

# Chat input at bottom
query = st.chat_input("Ask something about the PDFs...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    if not st.session_state.chunks:
        answer = "⚠️ Please upload at least one PDF first."
    else:
        context = retrieve(query, st.session_state.chunks, st.session_state.embeddings)
        answer = generate_answer(st.session_state.system_prompt, query, context)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
