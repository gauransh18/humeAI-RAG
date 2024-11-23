import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_local_websocket():
    # uri = "wss://humeai-rag.onrender.com/llm"
    uri = "ws://localhost:8000/llm"
    logger.info(f"Connecting to {uri}")
    
    extra_headers = {
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://humeai-rag.onrender.com"
    }
    
    try:
        async with websockets.connect(
            uri,
            extra_headers=extra_headers,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10
        ) as websocket:
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
            
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    asyncio.run(test_local_websocket())