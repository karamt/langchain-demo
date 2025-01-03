import os
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma:2b")

prompt = input("Enter a question.")
response = llm.invoke(prompt)
print(response.content)