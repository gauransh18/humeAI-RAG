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
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize Chroma
CHROMA_PATH = "chroma"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Initialize the language model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
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

# Initialize the QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
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

Helpful Answer: """,
            input_variables=["context", "question"]
        )
    }
)

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
                response = await asyncio.to_thread(
                    lambda: qa_chain({
                        "question": user_text,
                        "chat_history": []  # Always start fresh
                    })
                )
                
                if not response.get('answer'):
                    answer = "I don't have enough information to answer that question."
                else:
                    answer = response['answer'].strip()
                
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
    # Verify the database exists and is populated
    collection = db._collection
    logger.info(f"Total documents in database: {collection.count()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        