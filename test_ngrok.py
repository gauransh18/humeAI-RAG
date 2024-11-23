import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ngrok_websocket():
    uri = "wss://171a-36-255-16-54.ngrok-free.app/llm"
    logger.info(f"Connecting to {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected")
            
            message = {
                "type": "user_input",
                "text": "Who is andrew huberman?"
            }
            
            logger.info(f"Sending: {message}")
            await websocket.send(json.dumps(message))
            
            while True:
                response = await websocket.recv()
                logger.info(f"Received: {response}")
                
                response_data = json.loads(response)
                if response_data.get("type") == "assistant_end":
                    break
            
    except Exception as e:
        logger.error(f"Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ngrok_websocket())