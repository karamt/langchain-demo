import os

import numpy
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

text_1 = input("Enter a text")
text_2 = input("Enter another text")

response_1 = llm.embed_query(text_1)
response_2 = llm.embed_query(text_2)

similarity_score = numpy.dot(response_1,response_2)

print(similarity_score)
