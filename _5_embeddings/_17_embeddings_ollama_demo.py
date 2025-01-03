import os
from langchain_ollama import OllamaEmbeddings

llm = OllamaEmbeddings(model="mistral")

prompt = input("Enter a text.")
response = llm.embed_query(prompt)
print(response)