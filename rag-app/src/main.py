import streamlit as st
from loader import load_text
from cleaner import clean_text
from splitter import split_text
from embedder import Embedder
from vectordb import VectorDB
from retriever import Retriever
from qa import QA

# ----------------------------------------
# UI CONFIG (Full white mode)
# ----------------------------------------
st.set_page_config(page_title="RAG QA Chat", layout="wide")

st.markdown(
    """
    <style>

        /* MAIN BACKGROUND */
        body, .main, .block-container {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }

        /* All text always black */
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #000000 !important;
        }

        /* CHAT INPUT BOX (MOST IMPORTANT STYLE) */
        .stTextInput > div > div > input {
            background-color: #FFFFFF !important;     /* Pure white */
            color: #000000 !important;                /* Black text */
            border: 2px solid #D35400 !important;      /* Rust border */
            border-radius: 12px !important;            /* Smooth round */
            padding: 10px 14px !important;             /* Better spacing */
            box-shadow: 0px 3px 6px rgba(0,0,0,0.15) !important; /* Premium shadow */
            font-size: 16px !important;
        }

        /* INPUT BOX on focus (when typing) */
        .stTextInput > div > div > input:focus {
            border: 2px solid #E67E22 !important;       /* Light rust border */
            box-shadow: 0px 4px 8px rgba(214, 84, 0, 0.4) !important;
        }

        /* TEXTAREA (If you use it) */
        .stTextArea > div > textarea {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 2px solid #D35400 !important;
            border-radius: 12px !important;
            padding: 12px !important;
            box-shadow: 0px 3px 6px rgba(0,0,0,0.15) !important;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 2px solid #D35400 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #000000 !important;
        }

        /* BUTTONS */
        .stButton button {
            background-color: #D35400 !important;
            color: #FFFFFF !important;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 8px;
            border: none;
        }

        .stButton button:hover {
            background-color: #E67E22 !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------
# RAG Pipeline
# ----------------------------------------
def run_rag_pipeline(file_path):
    raw = load_text(file_path)
    clean = clean_text(raw)

    chunks = split_text(clean)
    st.write(f"📄 Total Chunks Created: {len(chunks)}")

    embedder = Embedder()
    chunk_embeddings = embedder.embed(chunks)

    db = VectorDB(dim=len(chunk_embeddings[0]))
    db.add(chunk_embeddings, chunks)

    retriever = Retriever(embedder, db)
    qa = QA()

    return retriever, qa


# ----------------------------------------
# MAIN UI
# ----------------------------------------
def main():

    st.title("🧠 RAG Chat — Ask Questions From Your Document")

    # Initialize states
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if "qa" not in st.session_state:
        st.session_state.qa = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    # ---------------------------
    # FILE UPLOAD
    # ---------------------------
    uploaded_file = st.file_uploader("📂 Upload a .txt document", type="txt")

    if uploaded_file and st.button("🚀 Process Document"):

        with open("uploaded_file.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):
            retriever, qa = run_rag_pipeline("uploaded_file.txt")

        st.session_state.retriever = retriever
        st.session_state.qa = qa

        st.success("Document processed! Ask your questions below 👇")


    # ---------------------------
    # CHAT INPUT + ANSWERS
    # ---------------------------
    if st.session_state.retriever and st.session_state.qa:

        st.subheader("💬 Ask a Question")

        # Text input behaves like ChatGPT box
        query = st.text_input("Type your question here:", key="user_question")

        if st.button("Submit Question") and query.strip():

            with st.spinner("Thinking..."):
                results = st.session_state.retriever.retrieve(query, top_k=2)
                ctx = " ".join([r[0] for r in results])
                answer = st.session_state.qa.answer(query, ctx)

            # Save to chat history
            st.session_state.chat_history.append((query, answer))


        # ---------------------------
        # DISPLAY CHAT HISTORY (Main window)
        # ---------------------------
        for q, a in st.session_state.chat_history:
            st.markdown(f"### 🟦 You: {q}")
            st.markdown(f"**🟩 Answer:** {a}")
            st.write("---")

    # ---------------------------
    # SIDEBAR CHAT HISTORY
    # ---------------------------
    st.sidebar.title("📝 Chat History")
    for q, a in st.session_state.chat_history:
        st.sidebar.write(f"**Q:** {q}")
        st.sidebar.write(f"**A:** {a}")
        st.sidebar.write("---")


    if st.button("Exit"):
        st.stop()


if __name__ == "__main__":
    main()
