
# 📘 RAG QA Chat Application

A **Retrieval-Augmented Generation (RAG)** based chat system that processes documents and answers user-generated questions using embeddings + a language model.
This project demonstrates how to combine text preprocessing, vector search, and generative AI into a working QA chatbot.

1. Add the text file.
2. Click on process document.
3. Ask the question from document.

<img width="1811" height="588" alt="image" src="https://github.com/user-attachments/assets/b9f797c7-be24-4b10-b634-6240a7c4a168" />
<img width="1457" height="770" alt="image" src="https://github.com/user-attachments/assets/52331436-122f-4790-a645-0ffb7c291ca6" />
<img width="1917" height="736" alt="image" src="https://github.com/user-attachments/assets/f6efc499-4687-47db-952d-17efa1df7513" />



# 📘 RAG QA Chat Application

A **Retrieval-Augmented Generation (RAG)** based chat system that processes documents and answers user-generated questions using embeddings + a language model.
This project demonstrates how to combine text preprocessing, vector search, and generative AI into a working QA chatbot.

---

## 🚀 Overview

This project allows you to:

* Upload a document
* Process & split the document
* Store embeddings in a FAISS vector DB
* Ask questions and receive context-aware answers

The system uses:

* **SentenceTransformer** for embeddings
* **FLAN-T5** for generating answers

---

## 🤖 Models Used

### **1. SentenceTransformer** (`BAAI/bge-small-en`)

#### ✅ Strengths

* Fast and efficient for generating embeddings
* Works well across domains
* Produces high-quality sentence embeddings

#### ❌ Weaknesses

* Context understanding is shallow compared to larger models
* Limited by its pretraining data

---

### **2. FLAN-T5** (`google/flan-t5-small`)

#### ✅ Strengths

* Fine-tuned for QA and general instruction tasks
* Lightweight → ideal for real-time RAG
* Low memory usage

#### ❌ Weaknesses

* Smaller size reduces answer accuracy
* Limited context window

---

## 📂 File Structure & Descriptions

| File           | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `main.py`      | Streamlit application. Manages upload, processing, and chat UI. |
| `loader.py`    | Loads raw text from files.                                      |
| `cleaner.py`   | Cleans up extra spaces, newlines, unwanted characters.          |
| `splitter.py`  | Splits text into chunks for embedding.                          |
| `embedder.py`  | Generates embeddings using SentenceTransformer.                 |
| `vectordb.py`  | Stores embeddings and performs FAISS similarity search.         |
| `retriever.py` | Retrieves relevant chunks for a query.                          |
| `qa.py`        | Uses FLAN-T5 to generate answers from retrieved chunks.         |

---

## 🛠️ Getting Started

### ✔️ Prerequisites

* Python **3.8+**
* Packages from `requirements.txt`

---

## 📥 Installation

```bash
git clone <repository-url>
cd project1/rag-app/src
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start Streamlit:

```bash
streamlit run main.py
```

---

## 💬 Usage

1. Upload a `.txt` document
2. Click **Process Document**
3. Ask questions in the chat
4. Receive context-based answers

---

## 🔮 Future Enhancements

* Integrate larger LLMs for improved response accuracy
* Add PDF / DOCX / Web Page ingestion
* Add vector caching for faster repeated queries
* Add summarization and document-level insights
