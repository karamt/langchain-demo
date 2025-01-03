import os
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug

set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)
print(OPENAI_API_KEY)

prompt = input("Enter a question.")
response = llm.invoke(prompt)
print(response.content)