import streamlit as st
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
import base64
import os

def encode_image(image):
    return base64.b64encode(image.read()).decode()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can identify a landmark."),
        (
            "human",
            [
                {"type": "text", "text": "Return the landmark name"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm
st.title("Landmark Helper")

uploaded_file = st.file_uploader("Upload your file", type=["jpg", 'png'])

question = st.text_input("Ask a question about the landmark: ")

task = None
if question:
    image = encode_image(uploaded_file)
    response = chain.invoke({"image": image})
    task = question + response.content

prompt = hub.pull("hwchase17/react")

tools = load_tools(["wikipedia","ddg-search"])
agent = create_react_agent(llm,tools,prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

if task:
    response = agent_executor.invoke({"input": task + " without any explanation"})
    st.write(response["output"])