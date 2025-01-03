import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOllama(model="llama3.2")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system"," You are an astronaut. Answer any question related to Solar system. If question is not related to Solar system, say I dont know"),
        ("human","{question}")
    ]
)

st.title("Solar System Info")

question = st.text_input("Enter any question related to Solar system")

chain = prompt_template | llm

if input:
    response = chain.invoke({"question": question})
    st.write(response.content)