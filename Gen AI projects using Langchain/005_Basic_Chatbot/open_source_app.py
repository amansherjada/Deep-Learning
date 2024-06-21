# Import necessary modules and libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables for API keys
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Enable Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Define a chat prompt template for the assistant
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's questions"),
        ("user", "Question:{question}")
    ]
)

## Streamlit framework
# Set up the Streamlit application interface
st.title('Langchain Demo with Llama 3 LLM')
# Create a text input box for user input
input_text = st.text_input('Search the topic you want')

## Llama 3 LLM
# Initialize the Ollama model from Langchain Community with Llama 3
llm = Ollama(model='llama3') # Make sure Llama 3 LLM is installed locally
# Initialize the output parser for processing the LLM's response
output_parser = StrOutputParser()

# Define the processing chain: prompt -> llama 3 model -> output parser
chain = prompt | llm | output_parser

# Check if there is user input; if so, invoke the processing chain and display the output
if input_text:
    st.write(chain.invoke({'question': input_text}))
