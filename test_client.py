import asyncio
import websockets
import json
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def test_evi():
    config_id = os.getenv('HUME_CONFIG_ID')
    api_key = os.getenv('HUME_API_KEY')
    
    uri = f"wss://api.hume.ai/v0/assistant/chat?config_id={config_id}&api_key={api_key}"
    logger.info(f"Connecting to Hume API")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to Hume API")
            
            test_message = {
                "type": "user_input",
                "text": "Tell me about machine learning"
            }
            
            logger.info(f"Sending: {test_message}")
            await websocket.send(json.dumps(test_message))
            
            while True:
                response = await websocket.recv()
                logger.info(f"Received: {response}")
                
                response_data = json.loads(response)
                if response_data.get("type") == "assistant_end":
                    break

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_evi())