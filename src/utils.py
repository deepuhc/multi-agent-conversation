"""
utils.py

This module provides utility functions for retrieving API keys from environment variables.
It uses the `dotenv` library to load keys from a `.env` file, ensuring secure access to
sensitive credentials.

Usage:
    Import and call the desired API key retrieval function (e.g., `get_gemini_api_key()`).
"""
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

def get_openai_api_key() -> str:
    """
    Retrieves the OpenAI API key from environment variables.

    Returns:
        str: The OpenAI API key.

    Raises:
        ValueError: If the API key is not found in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return api_key

def get_deepseek_api_key() -> str:
    """
    Retrieves the DeepSeek API key from environment variables.

    Returns:
        str: The DeepSeek API key.

    Raises:
        ValueError: If the API key is not found in the environment.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    return api_key

def get_gemini_api_key() -> str:
    """
    Retrieves the Gemini API key from environment variables.

    Returns:
        str: The Gemini API key.

    Raises:
        ValueError: If the API key is not found in the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return api_key