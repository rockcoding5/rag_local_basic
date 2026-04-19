# 🧠 Local RAG (Retrieval-Augmented Generation) — From Scratch

A simple, fully local RAG pipeline built using:

* Local LLM (via Ollama)
* Open-source embeddings (HuggingFace)
* Vector search (FAISS)
* Orchestration (LangChain)

👉 No API keys
👉 Runs entirely on your machine
👉 Built for learning core RAG concepts

---

# 🚀 What is RAG?

**Retrieval-Augmented Generation (RAG)**:

> Fetch relevant data → give it to LLM → generate grounded answer

---

# 🧩 Flow Diagram
![Flow Diagram](docs/rag_clean_sample.png)

---

# 🔍 Concepts Explained

## 📄 Document Loading

* Load PDF → convert to text
* Tool: PyPDFLoader

---

## ✂️ Chunking

* Split large text into smaller pieces
* Improves retrieval accuracy

---

## 🔢 Embeddings

* Convert text → vectors
* Enables semantic similarity

Example:

```text
"AI is powerful" → [0.21, -0.67, ...]
```

---

## 🗄️ Vector Database

* Stores embeddings
* Performs similarity search

👉 Using FAISS (fast + local)

---

## 🔎 Retrieval

* Finds top-K relevant chunks
* Based on vector similarity

---

## 🧱 Context Building

* Merge retrieved chunks
* Feed into LLM

---

## ✍️ Prompting

* Controls LLM behavior
* Prevents hallucination

---

## 🤖 LLM (Local)

* Generates final answer

Models:

* `phi3` → fast ⚡
* `mistral` → better reasoning 🧠

---

## ⏱️ Observability

* Time tracking per step
* Helps identify bottlenecks

---

# 🧪 Sample Input & Output

## 🔍 Input Query

```text
Give a summary of the program, its structure, and what it includes
```

---

## 📥 Retrieved Chunks (Example)

```text
Chunk 1:
The academy is divided into four phases...

Chunk 2:
Each track includes workshops, codelabs, assessments...
```

---

## 📤 Output Answer

```text
The program is structured into four phases including multiple cohorts
and a final hackathon. It includes workshops, hands-on codelabs,
optional skills labs, assessments, and project-based learning.

Participants collaborate on problem statements and build solutions
through guided challenge-based learning.
```

---

## ⏱️ Performance Example

```text
Document Loading: 1.2 sec
Embeddings: 7.4 sec
Vector DB: 1.5 sec
Retrieval: 0.05 sec
LLM Inference: 80 sec

Total Time: ~90 sec
```

---

# 🛠️ Tech Stack

| Layer      | Tool        |
| ---------- | ----------- |
| LLM        | Ollama      |
| Embeddings | HuggingFace |
| Vector DB  | FAISS       |
| Framework  | LangChain   |
| Loader     | PyPDF       |

---

# 💻 Local Setup

## 1. Install Python (3.11 recommended)

👉 https://www.python.org/downloads/

✔ Add to PATH

---

## 2. Install uv

```bash
pip install uv
```

---

## 3. Create project

```bash
mkdir rag-local
cd rag-local
uv init
```

---

## 4. Virtual environment

```bash
uv venv
.\.venv\Scripts\activate
```

---

## 5. Install dependencies

```bash
uv pip install langchain langchain-community langchain-core \
langchain-text-splitters langchain-huggingface \
faiss-cpu pypdf sentence-transformers
```

---

## 6. Install Ollama

👉 https://ollama.com

---

## 7. Pull model

```bash
ollama pull phi3
```

---

## 8. Add PDF

```text
FAQ_GEN_AI_APAC_EDITION.pdf
```

---

## 9. Run

```bash
python rag_local_basic.py
```

---

# ⚠️ Limitations

* Slow on CPU (no GPU)
* Basic retrieval (no re-ranking)
* PDF parsing can be messy

---

# 🔥 Future Improvements

* Multi-query retrieval
* Re-ranking
* Hybrid search
* Persistent vector DB
* Agentic workflows

---

# 🧠 Key Learnings

* Retrieval > Model
* Chunking matters more than expected
* Query quality changes everything
* LLM is the slowest part

---

# 🎯 Final Thought

> RAG is not magic.
> It’s controlled context + reasoning.

---

# ⭐ Star this repo

If this helped you — build on it 🚀
