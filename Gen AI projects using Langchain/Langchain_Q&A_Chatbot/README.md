# Basic Q&A Chatbot

This repository contains a basic Q&A chatbot built using the Fireworks AI model `bleat-adapter` and Streamlit for the user interface.

## Features

- Uses the Fireworks AI `bleat-adapter` model for generating responses.
- Streamlit-based web interface for easy interaction.
- Simple and intuitive user experience for asking and receiving answers to questions.

## Setup

### Prerequisites

- Python 3.7 or higher
- An API key for Fireworks AI

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/basic-qa-chatbot.git
    cd basic-qa-chatbot
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up your environment variables:
    - Create a `.env` file in the root directory of your project.
    - Add your Fireworks API key to the `.env` file:
      ```
      FIREWORKS_API_KEY=your_fireworks_api_key_here
      ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Hugging Face Space

You can also try out the chatbot on Hugging Face Spaces: [Langchain_Q-A_Chatbot](https://huggingface.co/spaces/amansherjada/Langchain_Q-A_Chatbot)
