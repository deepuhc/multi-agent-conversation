"""
MultiAgentConversation.py

This module implements a multi-agent conversation system using the AutoGen library, simulating
dialogues between historical figures (e.g., astronomers Ptolemy and Aryabhata). It supports
dynamic agent configuration via YAML, topic tracking, and robust error handling for professional
use. The conversation logic is encapsulated in a reusable ConversationManager class.

Usage:
    Run this script to initiate a conversation between configured agents, display chat history,
    cost, and summary. Configure agents in `config.yaml`.
"""
import yaml
import json
import time
from typing import Dict, List, Optional
from src.utils import get_gemini_api_key
import google.generativeai as genai
from autogen import ConversableAgent
import pprint
import backoff

class ConversationManager:
    """Manages multi-agent conversations with topic tracking and error handling."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ConversationManager with agent configurations from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.agents: Dict[str, ConversableAgent] = {}
        self.topic_history: List[Dict[str, str]] = []
        self.config_path = config_path
        self.llm_config = self._configure_llm()
        self._load_agents()

    def _configure_llm(self) -> dict:
        """
        Configure the language model settings for AutoGen agents using the Gemini API.

        Returns:
            dict: Configuration dictionary with model details, API key, and token limit.
        """
        api_key = get_gemini_api_key()
        return {
            "config_list": [{
                "model": "gemini-1.5-flash",
                "api_key": api_key,
                "api_type": "google"
            }],
            "max_tokens": 50
        }

    def _load_agents(self) -> None:
        """
        Load agent configurations from the YAML file and create ConversableAgent instances.

        Raises:
            FileNotFoundError: If the config file is not found.
            yaml.YAMLError: If the config file is invalid.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing {self.config_path}: {e}")

        for agent_config in config.get('agents', []):
            name = agent_config.get('name')
            system_message = agent_config.get('system_message')
            self.agents[name] = ConversableAgent(
                name=name,
                system_message=system_message,
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                is_termination_msg=lambda msg: "See you again later" in msg["content"],
            )

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def initiate_conversation(self, initiator_name: str, recipient_name: str, 
                             initial_message: str, max_turns: int = 2) -> dict:
        """
        Initiate a conversation between two agents and track the topic.

        Args:
            initiator_name (str): Name of the initiating agent.
            recipient_name (str): Name of the recipient agent.
            initial_message (str): Initial message to start the conversation.
            max_turns (int): Maximum number of conversation turns.

        Returns:
            dict: Result of the conversation, including chat history and cost.

        Raises:
            KeyError: If initiator or recipient name is not found in agents.
        """
        if initiator_name not in self.agents or recipient_name not in self.agents:
            raise KeyError("Initiator or recipient agent not found")

        initiator = self.agents[initiator_name]
        recipient = self.agents[recipient_name]
        
        # Track the initial topic
        self.topic_history.append({
            "initiator": initiator_name,
            "recipient": recipient_name,
            "topic": initial_message,
            "timestamp": time.time()
        })

        # Start the conversation with retry logic
        result = initiator.initiate_chat(
            recipient=recipient,
            message=initial_message,
            max_turns=max_turns,
            summary_method="reflection_with_llm",
            summary_prompt="Summarize the conversation",
        )
        return result

    def get_last_topic(self, agent_name: str) -> Optional[str]:
        """
        Retrieve the last topic discussed by the specified agent.

        Args:
            agent_name (str): Name of the agent to check.

        Returns:
            Optional[str]: The last topic discussed, or None if no history exists.
        """
        for topic in reversed(self.topic_history):
            if topic["initiator"] == agent_name or topic["recipient"] == agent_name:
                return topic["topic"]
        return None

def main():
    """
    Main function to demonstrate a conversation between configured agents.
    """
    try:
        # Initialize conversation manager
        manager = ConversationManager()

        # Initiate conversation between Ptolemy and Aryabhata
        initial_message = (
            "I'm Ptolemy. Aryabhata, what's your most interesting discovery?"
        )
        chat_result = manager.initiate_conversation(
            initiator_name="Ptolmey",
            recipient_name="Aryabhata",
            initial_message=initial_message,
            max_turns=2
        )

        # Simulate Aryabhata asking about the last topic
        last_topic = manager.get_last_topic("Aryabhata")
        if last_topic:
            manager.agents["Aryabhata"].send(
                message=f"What's the last topic we discussed? I recall: {last_topic}",
                recipient=manager.agents["Ptolmey"]
            )

        # Display results
        print("Chat History:")
        pprint.pprint(chat_result.chat_history)
        print("\nCost:")
        pprint.pprint(chat_result.cost)
        print("\nSummary:")
        pprint.pprint(chat_result.summary)
        print("\nTopic History:")
        pprint.pprint(manager.topic_history)

    except Exception as e:
        print(f"Error during conversation: {e}")

if __name__ == "__main__":
    main()