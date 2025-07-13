![App Screenshot](assets/chunkwise_UI_2.png)
![App Screenshot](assets/chunkwise_UI_1.png)

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

