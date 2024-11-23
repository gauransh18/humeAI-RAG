from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "docs/books"

def main():
    # Remove old database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize embeddings
    embeddings =  HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}  
    )
    # Load PDF documents
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    
    # Load text documents
    text_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    text_documents = text_loader.load()
    
    # Combine all documents
    all_documents = pdf_documents + text_documents
    
    if not all_documents:
        print("No documents found! Please add documents to the docs/books directory.")
        return
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.split_documents(all_documents)
    
    # Create and persist vector store
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    
    print(f"Processed {len(texts)} text chunks")
    
    # Test a query
    results = db.similarity_search("Who is Andrew Huberman?", k=2)
    print("\nTest query results:")
    for doc in results:
        print("\nContent:", doc.page_content[:200])

if __name__ == "__main__":
    main()
