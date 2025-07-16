from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

import os

# Load and split documents
def load_docs():
    loaders = [PyPDFLoader(f'docs/{f}') for f in os.listdir('docs') if f.endswith('.pdf')]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Build vector store
docs = load_docs()
embedding = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embedding)

# Setup retrieval-based QA chain with Ollama
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3"),
    retriever=retriever
)

def ask_question(query):
    return qa.run(query)
