from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import asyncio 
import re

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize Chroma
CHROMA_PATH = "chroma"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Initialize the language model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_TOKEN'],
    model_kwargs={
        "temperature": 0.7,
        "max_length": 512,
        "top_p": 0.9
    }
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
)

# Initialize the QA chain with more explicit configuration
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Explicitly set number of documents to retrieve
    ),
    memory=memory,
    return_source_documents=True,
    chain_type="stuff",
    verbose=True,
    combine_docs_chain_kwargs={
        "prompt": PromptTemplate(
            template="""You are a helpful assistant that provides accurate information about Andrew Huberman and his work.
Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know.

Context: {context}

Question: {question}

Important: Always provide a complete, informative answer based on the context provided.

Helpful Answer: """,
            input_variables=["context", "question"]
        )
    }
)

def preprocess_context(context: str) -> str:
    """Clean and structure the context for better RAG performance."""
    # Remove multiple newlines and spaces
    context = re.sub(r'\s+', ' ', context)
    # Remove special characters
    context = re.sub(r'[^\w\s.,?!-]', '', context)
    return context.strip()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.websocket("/llm")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("Received WebSocket connection request")
    
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                logger.info(f"Received raw message: {data}")
                
                # Parse the message
                message_data = json.loads(data)
                
                # Extract user text from the new message format
                user_text = ""
                if "messages" in message_data:
                    messages = message_data.get("messages", [])
                    for msg in messages:
                        if msg.get("type") == "user_message":
                            user_text = msg.get("message", {}).get("content", "")
                            break
                
                if not user_text:
                    logger.warning("No valid user text found in message")
                    continue
                
                logger.info(f"Extracted user text: {user_text}")
                
                # Create a fresh chain for each question
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=db.as_retriever(),
                    memory=ConversationBufferMemory(
                        memory_key="chat_history",
                        output_key="answer",
                        return_messages=True
                    ),
                    return_source_documents=True,
                    combine_docs_chain_kwargs={
                        "prompt": PromptTemplate(
                            template="""You are a helpful assistant that provides accurate information about Andrew Huberman and his work.
Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know.

Context: {context}

Question: {question}

Helpful Answer: """,
                            input_variables=["context", "question"]
                        )
                    }
                )
                
                # Get response from RAG system
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: qa_chain.invoke({
                            "question": user_text,
                            "chat_history": []
                        })
                    ),
                    timeout=30
                )
                
                # Add detailed logging
                logger.info(f"Raw chain result: {result}")
                logger.info(f"Retrieved documents: {[doc.page_content[:200] for doc in result.get('source_documents', [])]}")
                
                if result and 'answer' in result:
                    answer = result['answer'].strip()
                    if answer == "assistant":  # Catch the specific error case
                        logger.error("Model returned default 'assistant' response")
                        answer = "I apologize, but I'm having trouble accessing the information. Please try again."
                    else:
                        logger.info(f"Successfully generated answer: {answer}")
                else:
                    logger.warning("No answer in response")
                    answer = "I apologize, but I'm having trouble generating a response. Please try again."
                    
                logger.info(f"Question: {user_text}")
                logger.info(f"Generated answer: {answer}")
                
                # Send response
                evi_response = {
                    "type": "assistant_input",
                    "text": answer,
                    "prosody": {
                        "rate": "medium",
                        "pitch": "medium",
                        "volume": "medium"
                    }
                }
                await websocket.send_text(json.dumps(evi_response))
                await websocket.send_text(json.dumps({"type": "assistant_end"}))
                
            except WebSocketDisconnect:
                logger.info("Client disconnected gracefully")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                logger.exception("Full traceback:")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        logger.exception("Full traceback:")
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.on_event("startup")
async def startup_event():
    # Enhanced database verification
    collection = db._collection
    doc_count = collection.count()
    logger.info(f"Total documents in database: {doc_count}")
    
    # Sample a document to verify content
    if doc_count > 0:
        sample_results = db.similarity_search("andrew huberman", k=1)
        if sample_results:
            logger.info(f"Sample document content: {sample_results[0].page_content[:200]}...")
        else:
            logger.error("No documents found in similarity search")
    else:
        logger.error("Database appears to be empty")

# Add a health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test the database
        docs = db.similarity_search("test", k=1)
        # Test the model with a simple query
        test_response = await asyncio.to_thread(
            lambda: llm("Say 'healthy'")
        )
        return {
            "status": "healthy",
            "database_docs": len(docs),
            "model_test": test_response
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        