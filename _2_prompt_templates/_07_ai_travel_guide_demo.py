import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)
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
month_of_year = ["Jan","Feb","Mar","April","May","June","July","Aug","Sept","Oct","Nov","Dec"]
budget_list = ["low","Medium","High"]

st.title("Travel Guide")
city = st.text_input("Enter a City you want to visit")
month = st.selectbox("Select month",month_of_year)
language = st.selectbox("Select output language",["Hindi","English"])
budget = st.selectbox("Travel Budget", budget_list)
if city and month and language and budget:
    response = llm.invoke(prompt_template.format(city=city,
                                                 month=month,
                                                 language=language,
                                                 budget=budget))
    st.write(response.content)