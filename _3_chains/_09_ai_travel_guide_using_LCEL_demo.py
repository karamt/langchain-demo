import os
import streamlit as st
import calendar
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOllama(model="llama3.2")
prompt_template = PromptTemplate(
    input_variables=["city","month","language","budget"],
    template="""
        Welcome to the {city} travel guide!
            If you're visiting in {month}, here's what you can do:
            1. Must-visit attractions.
            2. Local cuisine you must try.
            3. Useful phrases in {language}.
            4. Tips for traveling on a {budget} budget.
        Enjoy your trip!
    """
)
# month_of_year = ["Jan","Feb","Mar","April","May","June","July","Aug","Sept","Oct","Nov","Dec"]
budget_list = ["low","Medium","High"]

st.title("Travel Guide")
city = st.text_input("Enter a City you want to visit")
month = st.selectbox("Select month",calendar.month_name[1:])
language = st.selectbox("Select output language",["Hindi","English"])
budget = st.selectbox("Travel Budget", budget_list)

chain = prompt_template | llm

chain_input = {"city":city,
                "month":month,
                "language":language,
                "budget":budget}

chain_input_alt = dict(city=city,
                       month=month,
                       language=language,
                       budget=budget)
print(type(chain_input))
if city and month and language and budget:
    response = chain.invoke(chain_input)
    st.write(response.content)