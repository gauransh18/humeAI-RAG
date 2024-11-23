from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"

def test_query(query: str):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load the existing database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Test search
    results = db.similarity_search_with_relevance_scores(query, k=2)
    
    print(f"\nQuery: {query}")
    print("\nResults:")
    for doc, score in results:
        print(f"\nRelevance score: {score}")
        print(f"Content: {doc.page_content[:200]}")

if __name__ == "__main__":
    test_query("Who is Andrew Huberman?")