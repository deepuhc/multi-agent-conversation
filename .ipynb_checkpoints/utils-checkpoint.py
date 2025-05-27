import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")

def get_deepseek_api_key():
    return os.getenv("DEEPSEEK_API_KEY")

def get_gemini_api_key():
    return os.getenv("GEMINI_API_KEY")