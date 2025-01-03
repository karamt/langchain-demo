import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)
prompt_title = PromptTemplate(
    input_variables=["topic"],
    template="""
        You are an experienced speech writer.
        You need to craft an impactful title for a speech
        on the following topic: {topic}
        Answer exactly with one title.
    """
)

prompt_speech = PromptTemplate(
    input_variables=["title"],
    template="""
        You need to write a powerful speech of 350 words
        for the following title: {title}
    """
)

first_chain = prompt_title | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])
print(first_chain)
second_chain = prompt_speech | llm
final_chain = first_chain | second_chain

st.title("Speech Generator")
topic = st.text_input("Enter a topic for speech")

if topic:
    response = final_chain.invoke(dict(topic=topic))
    st.write(response.content)