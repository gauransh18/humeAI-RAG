import json
from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import inflect
import re
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()


class Agent:
    """
    A chat agent class responsible for handling and processing chat messages to generate
    responses using a language model and additional tools.

    This class initializes with a system prompt to define the agent's personality or context,
    loads necessary tools for enhancing responses (like web search capabilities), and sets up
    a language model for generating chat responses.

    Attributes:
        system_prompt (str): A prompt that defines the initial context or personality of the agent.
        agent_executor (AgentExecutor): An executor to manage the agent's response generation process,
                                        including interaction with tools and language models.

    Methods:
        add_prosody_to_utterance(utterance: str, prosody: dict) -> str:
            Enhances an utterance with prosody information.

        parse_hume_message(messages_payload: dict) -> tuple[str, list]:
            Parses incoming messages and extracts necessary information for response generation.

        get_response(message: str, chat_history: list = None) -> str:
            Generates a response based on the given message and chat history.

        number_to_words(number: str) -> str:
            Converts numerical strings within the response to their word equivalents.
    """

    def __init__(self, *, system_prompt: str):
        """
        Initializes the agent with a given system prompt and sets up necessary components for
        response generation, including tools and a language model.

        Args:
            system_prompt (str): A string that sets the initial context or personality of the agent.
        """
        # Store the system prompt that will define the initial conversation context or personality for the agent.
        self.system_prompt = system_prompt

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}  
        )
        
        # Initialize vector store
        self.db = Chroma(
            persist_directory="chroma",
            embedding_function=self.embeddings
        )
        
        # Initialize language model
        self.llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-small",
            huggingfacehub_api_token=os.environ['HUGGINGFACE_API_TOKEN'],
            max_new_tokens=250,
            temperature=0.5,
            model_kwargs={"max_length": 512}
        )

    def add_prosody_to_utterance(self, utterance: str, prosody: dict) -> str:
        """
        Enhances an utterance by appending prosody information derived from prosody analysis.

        Args:
            utterance (str): The original text utterance to be enhanced.
            prosody (dict): A dictionary containing prosody features and their values.

        Returns:
            str: The enhanced utterance with prosody information appended.
        """

        prosody_string = ", ".join(prosody.keys())
        return f"Speech: {utterance} {prosody_string}"

    def parse_hume_message(self, messages_payload: dict) -> [str, list[any]]:
        """
        Parses the payload of messages received from a client, extracting the latest user message
        and constructing the chat history with contextualized utterances.

        Args:
            messages_payload (dict): The payload containing messages from the chat.

        Returns:
            tuple[str, list]: A tuple containing the last user message and the constructed chat history.
        """

        messages = messages_payload.get("messages", [])
        if not messages:
            raise ValueError("No messages found in the payload")

        last_user_message = messages[-1]["message"]["content"]
        chat_history = [SystemMessage(content=self.system_prompt)]

        for message in messages[:-1]:
            message_object = message["message"]
            if message_object["role"] == "user":
                chat_history.append(HumanMessage(content=message_object["content"]))
            elif message_object["role"] == "assistant":
                chat_history.append(AIMessage(content=message_object["content"]))

        return last_user_message, chat_history

    def get_responses(self, message: str, chat_history=None) -> list[str]:
        """
        Generates responses to the user's message based on the current chat history and the
        capabilities of the integrated language model and tools.

        Args:
            message (str): The latest message from the user.
            chat_history (list, optional): The chat history up to this point. Defaults to None.

        Returns:
            list[str]: The stream of generated responses from the agent.
        """

        if chat_history is None:
            chat_history = []

        # Perform RAG
        results = self.retrieve_documents(message)
        
        if not results or results[0][1] < 0.3:
            context_text = ""
            response = "I apologize, but I couldn't find relevant information to answer your question."
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt = f"Context: {context_text}\n\nQuestion: {message}\n\nAnswer:"
            response = self.llm.invoke(prompt)

        # Convert numbers to words for better speech
        numbers = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", response)
        for number in numbers:
            words = self.number_to_words(number)
            response = response.replace(number, words, 1)

        # Format response for EVI2
        responses = []
        responses.append(json.dumps({
            "type": "assistant_input",
            "text": response,
            "prosody": {
                "rate": "medium",
                "pitch": "medium",
                "volume": "medium"
            }
        }))
        responses.append(json.dumps({"type": "assistant_end"}))

        return responses

    def retrieve_documents(self, query: str):
        """Perform RAG document retrieval"""
        results = self.db.similarity_search_with_relevance_scores(query, k=3)
        return results

    def number_to_words(self, number):
        """
        Converts a number in string format into its word representation. For example,
        it would convert "42" to "forty-two". Useful for making numerical
        data more readable in responses.

        Args:
            number (str): The number to convert, in string format.

        Returns:
            str: The word representation of the given number.
        """
        p = inflect.engine()
        words = p.number_to_words(number)
        return words


if __name__ == "__main__":
    agent = Agent(system_prompt="You are a helpful assistant.")
    print("\n".join(agent.get_responses("Hello")))
