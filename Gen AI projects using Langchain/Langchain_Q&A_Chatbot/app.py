#Q&A Chatbot
from langchain.llms import Fireworks
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

#Function to load Fireworks model and get response

def get_fireworks_response(question):
    llm = Fireworks(api_key = os.getenv("FIREWORKS_API_KEY"),
                    model_name = "accounts/fireworks/models/bleat-adapter",
                    temperature = 0.6)
    response = llm(question)
    return response

## Initialize streamlit

st.set_page_config(page_title="Basic Q&A Bot")
st.header("Basic Q&A Chatbot")

input = st.text_input("Input: ", key="input")
response = get_fireworks_response(input)

submit = st.button("Ask the Question")

# If ask button is called
if submit:
    st.subheader("The Response is")
    st.write(response)
else:
    print("Provide some Input")