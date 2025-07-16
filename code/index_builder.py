import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Set up paths
docs_path = os.path.join(os.path.dirname(__file__), '../docs')
index_path = os.path.join(os.path.dirname(__file__), 'faiss_index')

# Load docs
docs = []
for f in os.listdir(docs_path):
    path = os.path.join(docs_path, f)
    if f.endswith('.pdf'):
        loader = PyPDFLoader(path)
    elif f.endswith('.txt'):
        loader = TextLoader(path)
    else:
        continue
    docs.extend(loader.load())

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Use a small embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and save index
db = FAISS.from_documents(split_docs, embedding)
db.save_local(index_path)

print("✅ Vector index saved at:", index_path)
