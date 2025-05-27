"""
test_multiagent.py

This module contains unit tests for the multi-agent conversation project.
It tests utility functions, agent creation, and the ConversationManager class
to ensure reliability and correctness.

Usage:
    Run with `python -m unittest tests/test_multiagent.py` to execute all tests.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import yaml
from src.utils import get_gemini_api_key, get_openai_api_key, get_deepseek_api_key
from src.MultiAgentConversation import configure_llm, create_ptolmey_agent, create_aryabhata_agent, ConversationManager

class TestMultiAgentConversation(unittest.TestCase):
    """Test suite for the multi-agent conversation functionality."""

    @patch('src.utils.os.getenv')
    def test_get_gemini_api_key_success(self, mock_getenv):
        """Test that get_gemini_api_key returns the correct API key."""
        mock_getenv.return_value = "test_gemini_key"
        api_key = get_gemini_api_key()
        self.assertEqual(api_key, "test_gemini_key")
        mock_getenv.assert_called_once_with("GEMINI_API_KEY")

    @patch('src.utils.os.getenv')
    def test_get_gemini_api_key_missing(self, mock_getenv):
        """Test that get_gemini_api_key raises ValueError when key is missing."""
        mock_getenv.return_value = None
        with self.assertRaises(ValueError) as context:
            get_gemini_api_key()
        self.assertEqual(str(context.exception), "GEMINI_API_KEY not found in environment variables")

    @patch('src.utils.os.getenv')
    def test_get_openai_api_key_success(self, mock_getenv):
        """Test that get_openai_api_key returns the correct API key."""
        mock_getenv.return_value = "test_openai_key"
        api_key = get_openai_api_key()
        self.assertEqual(api_key, "test_openai_key")
        mock_getenv.assert_called_once_with("OPENAI_API_KEY")

    @patch('src.utils.os.getenv')
    def test_get_deepseek_api_key_success(self, mock_getenv):
        """Test that get_deepseek_api_key returns the correct API key."""
        mock_getenv.return_value = "test_deepseek_key"
        api_key = get_deepseek_api_key()
        self.assertEqual(api_key, "test_deepseek_key")
        mock_getenv.assert_called_once_with("DEEPSEEK_API_KEY")

    @patch('src.MultiAgentConversation.get_gemini_api_key')
    def test_configure_llm(self, mock_get_api_key):
        """Test that configure_llm returns the expected LLM configuration."""
        mock_get_api_key.return_value = "test_gemini_key"
        llm_config = configure_llm()
        expected_config = {
            "config_list": [{
                "model": "gemini-1.5-flash",
                "api_key": "test_gemini_key",
                "api_type": "google"
            }],
            "max_tokens": 50
        }
        self.assertEqual(llm_config, expected_config)

    @patch('src.MultiAgentConversation.ConversableAgent')
    @patch('src.MultiAgentConversation.configure_llm')
    def test_create_ptolmey_agent(self, mock_configure_llm, mock_conversable_agent):
        """Test that create_ptolmey_agent configures the agent correctly."""
        mock_configure_llm.return_value = {"config_list": [], "max_tokens": 50}
        mock_agent_instance = MagicMock()
        mock_conversable_agent.return_value = mock_agent_instance

        agent = create_ptolmey_agent()

        self.assertEqual(agent, mock_agent_instance)
        mock_conversable_agent.assert_called_once_with(
            name="Ptolmey",
            system_message="You are Greek Astronomer Ptolemy, known for your geocentric model of the universe.",
            llm_config={"config_list": [], "max_tokens": 50},
            human_input_mode="NEVER",
            is_termination_msg=mock_conversable_agent.call_args[1]["is_termination_msg"]
        )

    @patch('src.MultiAgentConversation.ConversableAgent')
    @patch('src.MultiAgentConversation.configure_llm')
    def test_create_aryabhata_agent(self, mock_configure_llm, mock_conversable_agent):
        """Test that create_aryabhata_agent configures the agent correctly."""
        mock_configure_llm.return_value = {"config_list": [], "max_tokens": 50}
        mock_agent_instance = MagicMock()
        mock_conversable_agent.return_value = mock_agent_instance

        agent = create_aryabhata_agent()

        self.assertEqual(agent, mock_agent_instance)
        mock_conversable_agent.assert_called_once_with(
            name="Aryabhata",
            system_message=(
                "Act as Indian astronomer Aryabhata, known for your contributions to mathematics and astronomy, "
                "including the approximation of pi and heliocentric ideas."
            ),
            llm_config={"config_list": [], "max_tokens": 50},
            human_input_mode="NEVER",
            is_termination_msg=mock_conversable_agent.call_args[1]["is_termination_msg"]
        )

    @patch('src.MultiAgentConversation.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.MultiAgentConversation.get_gemini_api_key')
    @patch('src.MultiAgentConversation.ConversableAgent')
    def test_conversation_manager_load_agents(self, mock_conversable_agent, mock_get_api_key, mock_open, mock_yaml_load):
        """Test that ConversationManager loads agents from YAML correctly."""
        mock_get_api_key.return_value = "test_gemini_key"
        mock_yaml_load.return_value = {
            'agents': [
                {'name': 'Ptolmey', 'system_message': 'You are Ptolemy.'},
                {'name': 'Aryabhata', 'system_message': 'You are Aryabhata.'}
            ]
        }
        mock_agent_instance = MagicMock()
        mock_conversable_agent.return_value = mock_agent_instance

        manager = ConversationManager(config_path="dummy_config.yaml")

        self.assertEqual(len(manager.agents), 2)
        self.assertIn("Ptolmey", manager.agents)
        self.assertIn("Aryabhata", manager.agents)
        self.assertEqual(mock_conversable_agent.call_count, 2)

    @patch('src.MultiAgentConversation.yaml.safe_load')
    @patch('builtins.open')
    def test_conversation_manager_missing_config(self, mock_open, mock_yaml_load):
        """Test that ConversationManager raises FileNotFoundError for missing config."""
        mock_open.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError) as context:
            ConversationManager(config_path="nonexistent.yaml")
        self.assertEqual(str(context.exception), "Configuration file nonexistent.yaml not found")

    @patch('src.MultiAgentConversation.yaml.safe_load')
    @patch('builtins.open')
    def test_conversation_manager_invalid_yaml(self, mock_open, mock_yaml_load):
        """Test that ConversationManager raises YAMLError for invalid YAML."""
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")
        with self.assertRaises(yaml.YAMLError) as context:
            ConversationManager(config_path="invalid.yaml")
        self.assertEqual(str(context.exception), "Error parsing invalid.yaml: Invalid YAML")

    @patch('src.MultiAgentConversation.get_gemini_api_key')
    def test_conversation_manager_get_last_topic(self, mock_get_api_key):
        """Test that get_last_topic retrieves the correct topic."""
        mock_get_api_key.return_value = "test_gemini_key"
        manager = ConversationManager(config_path="config.yaml")
        manager.topic_history = [
            {"initiator": "Ptolmey", "recipient": "Aryabhata", "topic": "Discovery", "timestamp": 12345.0},
            {"initiator": "Aryabhata", "recipient": "Ptolmey", "topic": "Follow-up", "timestamp": 12346.0}
        ]
        topic = manager.get_last_topic("Aryabhata")
        self.assertEqual(topic, "Follow-up")

    @patch('src.MultiAgentConversation.get_gemini_api_key')
    @patch('src.MultiAgentConversation.ConversableAgent')
    def test_initiate_conversation_invalid_agent(self, mock_conversable_agent, mock_get_api_key):
        """Test that initiate_conversation raises KeyError for invalid agent names."""
        mock_get_api_key.return_value = "test_gemini_key"
        manager = ConversationManager(config_path="config.yaml")
        with self.assertRaises(KeyError) as context:
            manager.initiate_conversation("InvalidAgent", "Aryabhata", "Hello")
        self.assertEqual(str(context.exception), "'Initiator or recipient agent not found'")

if __name__ == '__main__':
    unittest.main()