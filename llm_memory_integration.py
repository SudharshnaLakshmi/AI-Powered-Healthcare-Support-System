import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DATA_PATH = "data/"

def load_pdf_files():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    #print(f"Loaded {len(documents)} pages")
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_store(text_chunks):
    os.makedirs("vectorstore", exist_ok=True)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local("vectorstore/db_faiss")
    #print("Vector store created.")

def main():
    documents = load_pdf_files()
    text_chunks = create_chunks(documents)
    create_vector_store(text_chunks)

if __name__ == "__main__":
    main()
