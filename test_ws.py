import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_local_websocket():
    # uri = "wss://humeevi-rag.onrender.com/llm" 
    uri = "ws://localhost:8000/llm"
    logger.info(f"Connecting to {uri}")
    
    async with websockets.connect(uri) as websocket:
        logger.info("Connected")
        
        message = {
            "messages": [{
                "type": "user_message",
                "custom_session_id": None,
                "message": {
                    "role": "user",
                    "content": "Who is Andrew Huberman?",
                    "tool_call": None,
                    "tool_result": None
                },
                "from_text": False
            }],
            "custom_session_id": None,
            "chat_id": "test-chat-id"
        }
        
        logger.info(f"Sending: {message}")
        await websocket.send(json.dumps(message))
        
        response = await websocket.recv()
        logger.info(f"Received: {response}")

if __name__ == "__main__":
    asyncio.run(test_local_websocket())