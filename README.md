Multi-Agent Conversation Simulator
Welcome to the Multi-Agent Conversation Simulator, a Python project that leverages the AutoGen library to simulate engaging dialogues between historical figures, such as astronomers Ptolemy and Aryabhata. This project showcases AI-driven multi-agent interactions, with features like dynamic agent configuration via YAML, topic history tracking, and robust error handling with retry logic. It’s designed to be modular, reusable, and ready for professional environments, complete with unit tests and clear documentation.
Features

Dynamic Agent Configuration: Define agents and their personas in a config.yaml file for easy customization.
Topic History Tracking: Keep track of conversation topics, enabling intelligent follow-up questions.
Robust Error Handling: Includes retry logic for API calls and validation for configuration files.
Unit Tests: Comprehensive tests using unittest to ensure reliability.
Professional Structure: Organized with a src/ directory for code, tests/ for unit tests, and clear documentation.

Prerequisites

Python: Version 3.10.13
Dependencies: Listed in requirements.txt
Gemini API Key: Required for running the main script (tests are API-independent)

Setup Instructions

Clone the Repository (once pushed to GitHub):
git clone https://github.com/deepuhc/multi-agent-conversation.git
cd multi-agent-conversation


Install Python 3.10.13:Use pyenv to install the correct Python version:
brew install pyenv
pyenv install 3.10.13
pyenv global 3.10.13
python3 --version  # Should show Python 3.10.13


Create and Activate a Virtual Environment:
python3 -m venv venv
source venv/bin/activate


Install Dependencies:
pip install -r requirements.txt


Set Up the Gemini API Key:Create a .env file in the project root:
touch .env
echo 'GEMINI_API_KEY=your_actual_gemini_api_key_here' > .env

Replace your_actual_gemini_api_key_here with your Gemini API key from Google's API Console.


Usage
Run the main script to start a conversation between Ptolemy and Aryabhata:
python src/MultiAgentConversation.py


The script initiates a conversation, tracks topics, and prints the chat history, cost, summary, and topic history.
Example output:Chat History:
[{'content': "I'm Ptolemy. Aryabhata, what's your most interesting discovery?", ...}, ...]
Cost:
{'total_cost': 0.0, ...}
Summary:
'Ptolemy and Aryabhata discussed their astronomical discoveries.'
Topic History:
[{'initiator': 'Ptolmey', 'recipient': 'Aryabhata', 'topic': "I'm Ptolemy...", 'timestamp': 12345.0}]



Run unit tests to verify functionality:
python -m unittest tests/test_multiagent.py -v

Project Structure
multi-agent-conversation/
├── src/
│   ├── MultiAgentConversation.py  # Core conversation logic
│   └── utils.py                  # Utility functions for API key retrieval
├── tests/
│   └── test_multiagent.py        # Unit tests
├── config.yaml                   # Agent configuration
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── README.md                     # This file

Configuration
Edit config.yaml to customize agents:
agents:
  - name: Ptolmey
    system_message: "You are Greek Astronomer Ptolemy, known for your geocentric model of the universe."
  - name: Aryabhata
    system_message: "Act as Indian astronomer Aryabhata, known for your contributions to mathematics and astronomy."

Add new agents by extending the agents list with name and system_message.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built with AutoGen for multi-agent interactions.
Uses Google's Gemini API for language model integration.

