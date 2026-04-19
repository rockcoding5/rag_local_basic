#Ingest → Split → Embed → Index → Retrieve → Prompt → Generate
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama

# =========================
# TIMER START
# =========================
total_start = time.time()

def log_time(step_name, start_time):
    elapsed = time.time() - start_time
    print(f"\n⏱️ {step_name} took: {elapsed:.2f} sec")
    return time.time()

# =========================
# 1. Load document
# =========================
start = time.time()

loader = PyPDFLoader("FAQ_GEN_AI_APAC_EDITION.pdf")
docs = loader.load()

# Clean PDF formatting
for doc in docs:
    doc.page_content = doc.page_content.replace("\n", " ")

start = log_time("Document Loading + Cleaning", start)

# =========================
# 2. Chunking
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

chunks = text_splitter.split_documents(docs)

print(f"\n📄 Total chunks created: {len(chunks)}")

start = log_time("Chunking", start)

# =========================
# 3. Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

start = log_time("Embedding Model Load", start)

# =========================
# 4. Vector DB Creation
# =========================
vectorstore = FAISS.from_documents(chunks, embeddings)

start = log_time("Vector DB Creation", start)

# =========================
# 5. Query
# =========================
query = "Give a summary of the program, its structure, and what it includes"

print(f"\n🔍 Query: {query}")

# =========================
# 6. Retrieval
# =========================
docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)

print("\n--- Retrieved Chunks with Scores ---\n")

for i, (doc, score) in enumerate(docs_with_scores):
    print(f"Chunk {i+1} | Score: {score:.4f}")
    print(doc.page_content[:200], "...\n")

start = log_time("Retrieval", start)

# =========================
# 7. Filter low-quality results
# =========================
filtered_docs = [doc for doc, score in docs_with_scores if score < 1.5]

if not filtered_docs:
    print("\n❌ No relevant context found. Exiting.")
    exit()

# =========================
# 8. Build context
# =========================
context = "\n\n---\n\n".join([doc.page_content for doc in filtered_docs])

start = log_time("Context Building", start)

# =========================
# 9. Prompt
# =========================
prompt = f"""Answer ONLY using the context.
If not found, say "I don't know".

Context:
{context}

Question:
{query}

Answer:"""

# =========================
# 10. LLM
# =========================
#llm = Ollama(model="mistral")
llm = Ollama(model="phi3", num_predict=200)

start = log_time("LLM Load", start)

response = llm.invoke(prompt)

start = log_time("LLM Inference", start)

# =========================
# 11. Output
# =========================
print("\n================ ANSWER ================\n")
print(response)

# =========================
# TOTAL TIME
# =========================
total_time = time.time() - total_start
print(f"\n🚀 Total Execution Time: {total_time:.2f} sec")
