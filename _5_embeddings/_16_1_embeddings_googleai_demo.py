import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

GOOGLE_API_KEY = "AIzaSyB-i9YL2_PzzmEIMBZTNoOc-wTJrqwGjcA"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = embeddings.embed_query("Hello")
