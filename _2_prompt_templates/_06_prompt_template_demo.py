import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)
prompt_template = PromptTemplate(
    input_variables=["country","number_of_paras","language"],
    template="""
    You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?  
    Answer in {number_of_paras} short paras in {language} language
    """
)

st.title("Cuisine Info")
country = st.text_input("Enter a country name to know about its dishes")
number_of_paras = st.number_input("Enter number of paragraphs (1 through 5)",min_value=1,max_value=5)
language = st.text_input("Enter language")
if country:
    response = llm.invoke(prompt_template.format(country=country,
                                                 number_of_paras=number_of_paras,
                                                 language=language))
    st.write(response.content)