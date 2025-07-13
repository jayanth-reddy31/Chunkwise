# streamlit_app.py

import streamlit as st
import os
import pickle
import numpy as np
import faiss
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import PrivateAttr
from bs4 import BeautifulSoup
import requests
from langchain.docstore.document import Document

# Set Groq API Key
#USE THIS WHEN YOUR API_KEY IS IN .env FILE
#from dotenv import load_dotenv
#load_dotenv() #take environment variables from .env

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# Streamlit UI
st.subheader("Made with ❤️ by Jayanth")
st.title("📚 Chunkwise")

# Sidebar for URL input
st.sidebar.title("Enter up to 3 URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())
urls = list(dict.fromkeys(urls))  # remove duplicates

process_button = st.sidebar.button("📥 Process URLs")

# Main area
if process_button and urls:
    with st.spinner("🔍 Loading and processing documents..."):
        
        

        def robust_url_loader(url):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                }
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator="\n")
                return Document(page_content=text.strip(), metadata={"source": url})
            except Exception as e:
                st.warning(f"⚠️ Failed to load {url}: {e}")
                return None
        
        # Load and filter valid docs
        docs = []
        for url in urls:
            doc = robust_url_loader(url)
            if doc and len(doc.page_content) > 100:
                st.success(f"✅ Loaded {url} | Length: {len(doc.page_content)} characters")
                docs.append(doc)
            else:
                st.warning(f"⚠️ No usable content from: {url}")
        
        if not docs:
            st.error("❌ None of the URLs produced valid content. Please try different links.")
            st.stop()

        for i, doc in enumerate(docs):
            st.write(f"🔹 Document {i+1} | Length: {len(doc.page_content)} characters")
            st.write(f"Source: {doc.metadata.get('source', 'N/A')}")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(docs)
        if len(docs) == 0:
            st.error("⚠️ No content loaded from the URLs. Make sure they are valid and accessible.")
            st.stop()
        texts = [doc.page_content for doc in docs]
        if not texts:
            st.error("No content extracted from the provided URLs. Please check the links.")
            st.stop()
        # Encode using SentenceTransformer
        encoder = SentenceTransformer("all-mpnet-base-v2")
        vectors = encoder.encode(texts)

        # Create FAISS index
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # Save for later use
        with open("documents.pkl", "wb") as f:
            pickle.dump(docs, f)
        faiss.write_index(index, "faiss_index.index")
        with open("encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)

        st.success("✅ URLs processed and indexed successfully!")


# Load components if available
if os.path.exists("faiss_index.index") and os.path.exists("documents.pkl") and os.path.exists("encoder.pkl"):
    index = faiss.read_index("faiss_index.index")
    with open("documents.pkl", "rb") as f:
        docs = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    # Define Custom FAISS Retriever
    class FAISSRetriever(BaseRetriever):
        _index: any = PrivateAttr()
        _docs: List[Document] = PrivateAttr()
        _encoder: any = PrivateAttr()
        _k: int = PrivateAttr(default=5)

        def __init__(self, index, docs, encoder, k=5):
            super().__init__()
            self._index = index
            self._docs = docs
            self._encoder = encoder
            self._k = k

        def get_relevant_documents(self, query: str) -> List[Document]:
            query_vec = self._encoder.encode([query])
            D, I = self._index.search(np.array(query_vec), self._k)
            return [self._docs[i] for i in I[0]]

        async def aget_relevant_documents(self, query: str) -> List[Document]:
            return self.get_relevant_documents(query)

    # Initialize Groq LLM
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

    retriever = FAISSRetriever(index=index, docs=docs, encoder=encoder)
    
    
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

    # Q&A Input
    query = st.text_input("❓ Ask a question based on the URLs above:")


    if query:
        with st.spinner("🤖 Generating answer..."):
            response = chain({"question": query}, return_only_outputs=True)
            st.subheader("📌 Answer:")
            st.write(response['answer'])

            if response.get("sources"):
                st.markdown("#### 🔗 Sources:")
                st.write(response["sources"])

st.write("For help, contact: jayshivareddy@gmail.com")
