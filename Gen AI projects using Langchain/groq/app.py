# Streamlit is an open-source app framework used for creating and sharing custom web apps for machine learning and data science.
import streamlit as st

# OS module provides functions to interact with the operating system, for example, to manage environment variables.
import os

# Langchain_groq is a library to interact with Groq language models.
from langchain_groq import ChatGroq

# Langchain_community provides community-contributed embedding models. OllamaEmbeddings is one such embedding model.
from langchain_community.embeddings import OllamaEmbeddings

# HuggingFaceEmbeddings is another embedding model from the langchain_community library, using Hugging Face models.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Langchain_text_splitters provides text splitting utilities, such as RecursiveCharacterTextSplitter for splitting documents into manageable chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Langchain.chains.combine_documents is used to combine multiple documents into one with a specific chain.
from langchain.chains.combine_documents import create_stuff_documents_chain

# Langchain_core.prompts provides utilities for creating and managing prompts, such as ChatPromptTemplate.
from langchain_core.prompts import ChatPromptTemplate

# Langchain.chains is used to create chains for document retrieval and other processes.
from langchain.chains import create_retrieval_chain

# Langchain.document_loaders provides document loading utilities, such as WebBaseLoader for loading documents from the web.
from langchain.document_loaders import WebBaseLoader

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Import FAISS vector store for managing vector embeddings
from langchain_community.vectorstores import FAISS

# Import time module for measuring response times
import time

# Load the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Streamlit session state if necessary
if "vector" not in st.session_state:
    try:
        # Attempt to use HuggingFaceEmbeddings
        st.session_state.embeddings = HuggingFaceEmbeddings()
    except ValueError:
        # Fallback to HuggingFaceEmbeddings in case of an error
        st.session_state.embeddings = HuggingFaceEmbeddings()

    # Load documents from a specified URL
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    # Split documents into manageable chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    # Create vector embeddings from the document chunks
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Inject CSS for the background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://miro.medium.com/v2/resize:fit:1100/format:webp/0*nOG7TYG49wdHRWdD");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit interface setup
st.title("ChatGroq Demo")

# Initialize the Groq language model chat interface
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define a chat prompt template
prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create a document chain for combining document chunks
document_chain = create_stuff_documents_chain(llm, prompt)

# Set up a retriever to fetch relevant documents
retriever = st.session_state.vectors.as_retriever()

# Create a retrieval chain for processing user queries
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Get user input from Streamlit
user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    response_time = time.process_time() - start
    print("Response time:", response_time)

    # Display the response
    st.write(response['answer'])

    # With a Streamlit expander, show document similarity search results
    with st.expander("Document Similarity Search"):
        # Find and display relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")