import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env
load_dotenv()

# Load documents (PDF + TXT)
def load_docs():
    docs_path = os.path.join(os.path.dirname(__file__), '../docs')
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Build vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# Setup online LLM using OpenRouter via LangChain
llm = ChatOpenAI(
    model_name="meta-llama/llama-3-8b-instruct",
    temperature=0.7
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(query):
    return qa.run(query)

# Example use
if __name__ == "__main__":
    print(ask_question("Summarize the documents."))
