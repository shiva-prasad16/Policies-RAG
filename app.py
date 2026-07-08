import streamlit as st
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Enterprise Document Assistant",
    page_icon="📄",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.block-container{
    padding-top:2rem;
}

.stButton>button{
    width:100%;
    height:50px;
    background:#2563eb;
    color:white;
    border-radius:10px;
    border:none;
    font-size:18px;
    font-weight:bold;
}

.answer-box{
    background:white;
    padding:25px;
    border-radius:12px;
    border-left:6px solid #2563eb;
    box-shadow:0px 3px 10px rgba(0,0,0,0.15);
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.title("📄 Enterprise Document Assistant")

st.caption(
    "Ask questions from HR, Leave, Insurance, Travel and Company Policies"
)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("# 📄 Enterprise AI")

    st.title("Enterprise AI")

    st.success("Ready")

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["NVIDIA_API_KEY"]
    )

    return embedding_model, client


embedding_model, client = load_models()

# ==========================================================
# LOAD INDEX
# ==========================================================

@st.cache_resource
def load_vector_store():

    index = faiss.read_index(
        "faiss_index.index"
    )

    chunks = np.load(
        "chunks.npy",
        allow_pickle=True
    )

    metadata = np.load(
        "metadata.npy",
        allow_pickle=True
    )

    return index, chunks, metadata

index, chunks, metadata = load_vector_store()

with st.sidebar:

    st.metric(
        "Vectors",
        index.ntotal
    )

    st.metric(
        "Chunks",
        len(chunks)
    )

# ==========================================================
# RETRIEVAL
# ==========================================================

def retrieve(query, k=3):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    retrieved = []

    for idx in indices[0]:

        retrieved.append({

            "text": chunks[idx],

            "source": metadata[idx]["source"]

        })

    return retrieved
# ==========================================================
# BUILD PROMPT
# ==========================================================

def build_prompt(question, retrieved):

    context = ""

    for item in retrieved:

        context += f"""
Source: {item['source']}

{item['text']}

----------------------------------------
"""

    prompt = f"""
You are an Enterprise HR Policy Assistant.

Answer ONLY using the information provided in the context.

Instructions:
- Give a clear and professional answer.
- Keep the answer concise.
- Do not repeat the context.
- Do not invent information.
- If the answer is not found, reply:
"The information is not available in the provided documents."

Context:

{context}

Question:
{question}

Answer:
"""

    return prompt


# ==========================================================
# CHAT HISTORY
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================================
# USER INPUT
# ==========================================================

question = st.chat_input(
    "Ask anything about your company policies..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # --------------------------------------
    # Retrieve Documents
    # --------------------------------------

    with st.spinner("Searching documents..."):

        retrieved = retrieve(question)

        prompt = build_prompt(
            question,
            retrieved
        )

    # --------------------------------------
    # Generate Answer
    # --------------------------------------

    with st.spinner("Generating answer..."):

        response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {
            "role": "system",
            "content": "You are an Enterprise HR Policy Assistant. Answer ONLY from the provided context. If the answer is not present, say that it is not available in the provided documents."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2,
    max_tokens=300
)

    answer = response.choices[0].message.content

    # --------------------------------------
    # Assistant Response
    # --------------------------------------

    with st.chat_message("assistant"):

        st.markdown(
            f"""
<div class="answer-box">

<h3>🤖 Answer</h3>

<p>{answer}</p>

</div>
""",
            unsafe_allow_html=True
        )

        # -------------------------------
        # Source Documents
        # -------------------------------

        st.markdown("### 📚 Source Documents")

        sources = []

        for item in retrieved:

            if item["source"] not in sources:
                sources.append(item["source"])

        for source in sources:

            st.success(source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": {answer}
        }
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Enterprise Document Assistant • Powered by Sentence Transformers + FAISS + FLAN-T5 + Streamlit"
)
