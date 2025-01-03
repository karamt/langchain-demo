import os
from langchain_openai import ChatOpenAI
import streamlit as st

st.title("Ask anything")

with st.sidebar:
    st.title("Provide your OPENAI API KEY")
    OPENAI_API_KEY = st.text_input("Enter your openai api key",type="password")
if not OPENAI_API_KEY:
    st.info("API key is needed to proceed")
    st.stop()

llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)

prompt = st.text_input("Enter a question:")
if prompt:
    response = llm.invoke(prompt)
    st.write(response.content)