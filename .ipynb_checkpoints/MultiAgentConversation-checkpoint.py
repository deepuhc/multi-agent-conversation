from utils import get_gemini_api_key
import google.generativeai as genai
from autogen import ConversableAgent
api_key = get_gemini_api_key()
llm_config = {
    "config_list": [{
        "model": "gemini-1.5-flash",
        "api_key": api_key,
        "api_type": "google"
    }],
    "max_tokens": 50
}

agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reply = agent.generate_reply(
    messages = [{"content": "Tell me a joke", "role": "user"}]
)
print(reply)