<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/jayanth-reddy31/Chunkwise/main/chunkwise_UI_1.png" width="600"/></td>
    <td><img src="https://raw.githubusercontent.com/jayanth-reddy31/Chunkwise/main/chunkwise_UI_2.png" width="600"/></td>
  </tr>
</table>



# 📚 URL-Based Q&A Application using LangChain, FAISS, Groq, and Streamlit

This project is an intelligent research assistant that enables users to input news article URLs and ask contextual questions. It processes content using **LangChain's RAG (Retrieval-Augmented Generation)** pipeline and answers questions by referencing specific article chunks. Designed for **finance analysts, researchers**, and **information seekers**, this tool overcomes traditional LLM limitations.

---

## 🔧 Technologies Used

| Component            | Technology                             |
|---------------------|-----------------------------------------|
| Frontend UI         | Streamlit                               |
| LLM Backend         | Groq (LLaMA3 API)                        |
| Vector Embeddings   | SentenceTransformer (`all-mpnet-base-v2`) |
| Vector Database     | FAISS (Facebook AI Similarity Search)   |
| Document Splitter   | RecursiveCharacterTextSplitter (LangChain) |
| Programming Language| Python                                  |

---

## 🚨 Why Not Just Use ChatGPT?

While ChatGPT and other LLMs are powerful, they fall short for real-time equity or news-based research due to:

- ❌ Tedious copy-pasting of long articles  
- ❌ No unified knowledge base for cross-article context  
- ❌ Token/word limits (typically ~3000 words)  
- ❌ Higher cost per token when feeding full documents  

---

## 🎯 What This Tool Solves

✅ Automates document fetching from URLs  
✅ Splits large texts into manageable **"chunks"**  
✅ Embeds and stores them in a searchable vector DB  
✅ Retrieves only relevant chunks for each question  
✅ Answers queries using a smart, token-efficient QA chain  
✅ Cites the **source URLs** in answers  

---

## 🧠 Core Concept: RAG (Retrieval-Augmented Generation)

This project uses **RAG** to power question-answering:

> Instead of feeding the entire document to the LLM, **RAG retrieves only relevant "chunks"** from stored embeddings and uses them as context for generating answers.

**Think of it like:**

> 📖 “Open book exam where you first look up relevant sections before answering the question.”

---

## 🔄 Pipeline Flow

![App Screenshot](https://raw.githubusercontent.com/jayanth-reddy31/Chunkwise/main/chunkwise_architecture.png)



---

## 📖 Key Concepts

### 🧩 What are Chunks?

- Chunks are **small segments of an article** (~500 words)  
- Required because LLMs can’t process entire long documents directly  
- Maintains semantic integrity via **smart splitting** and **overlapping** for context  

---

### ✂️ Chunking Strategies

#### 1. **CharacterTextSplitter**
- Splits text by character count and a chosen separator  
- ❌ May still produce oversized or broken chunks  

#### 2. ✅ **RecursiveCharacterTextSplitter (RECOMMENDED)**
- Splits text **recursively** by multiple priority separators  
  (e.g., `["\n\n", "\n", ".", " "]`)  
- Ensures chunks are under LLM token limits  
- Adds **chunk overlap** to preserve context  

---

### 🔗 Chunk Merging Logic

After splitting, adjacent chunks may be **merged** to optimize LLM input usage:

> Example:  
> - ch1: 1000 tokens  
> - ch2: 2000 tokens  
> → Merge to create a single 3000-token input (if within model limits)

---

## 🧮 Embeddings & Semantic Search

### 🔍 What are Embeddings?

- Embeddings are **768-dimensional numeric vectors**  
- Represent the **semantic meaning** of a sentence or chunk  
- Generated using `all-mpnet-base-v2` SentenceTransformer

### 📦 FAISS Vector DB

- **In-memory** vector store for fast similarity search  
- Ideal for small- to medium-scale applications  
- For production-scale: use Pinecone, Chroma, or Milvus  

---

## 🧵 Retrieval and QA Chain

### 🔁 How Retrieval Works:

1. User’s question → converted to embedding  
2. FAISS performs **vector similarity search**  
3. Returns top `k` relevant chunks  
4. Sends to LLM for **answer generation**

---

### 🧠 QA Methods

#### 1. **Stuff Method**
- All retrieved chunks are **stuffed into a single prompt**  
- ✅ Fast and simple  
- ❌ Breaks if token limit is exceeded  

#### 2. ✅ **MapReduce Method**
- LLM processes **each chunk independently**  
- Summarizes individual answers into a final response  
- ✅ Scalable for large inputs  
- ❌ Requires **multiple LLM calls** → more processing time  

---

## 🖥️ Streamlit UI Features

- 🔗 Accepts **up to 3 URLs**  
- 🧠 “📥 Process URLs” button initiates scraping, splitting, embedding, and indexing  
- ❓ Ask any question based on the articles  
- 📌 Displays the **answer + source citations**


