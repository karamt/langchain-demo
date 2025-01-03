import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)
#llm = ChatOllama(model="llama3.2")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system"," You are an astronaut. Answer any question related to Solar system. If question is not related to Solar system, say I dont know"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human","{question}")
    ]
)

chain = prompt_template | llm
history_for_chain = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id:history_for_chain,
    input_messages_key="question",
    history_messages_key="chat_history"
)

print("Solar System Info")

while True:
    question = input("Enter any question related to Solar system")

    if input:
        response = chain_with_history.invoke(
            {"question": question},
            {"configurable":{
                "session_id":"session1234"
            }})
        print(response.content)
