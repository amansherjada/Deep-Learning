from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Ensure the ANTHROPIC_API_KEY environment variable is set
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Create the FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# Initialize the ChatAnthropic instance with the required model_name
chat_anthropic = ChatAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model_name='claude-3-opus-20240229'
)

# Add routes to the FastAPI app
add_routes(
    app,
    chat_anthropic,  # Use the initialized ChatAnthropic instance
    path="/claudeai"
)

# Initialize the Ollama model
llm = Ollama(model="llama3")

# Define chat prompts
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5 years child with 100 words")

# Add routes for the prompts
add_routes(
    app,
    prompt1 | chat_anthropic,
    path="/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)