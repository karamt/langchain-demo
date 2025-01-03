import os
from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral")

prompt = input("Enter a question.")
response = llm.invoke(prompt)
print(response.content)