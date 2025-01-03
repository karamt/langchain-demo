import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import base64
import os


def encode_image(image):
    return base64.b64encode(image.read()).decode()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can analyze images of nutrition charts"
                   "and help choose the right diet"),
        (
            "human",
            [
                {"type": "text", "text": "{input}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image1}",
                        "detail": "low",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image2}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm
st.title("Diet Helper")

uploaded_file_1 = st.file_uploader("Upload first nutrition chart image", type=["jpg", 'png'])
uploaded_file_2 = st.file_uploader("Upload second nutrition chart image", type=["jpg", 'png'])

question = st.text_input("Ask a question about image")

if question:
    image1 = encode_image(uploaded_file_1)
    image2 = encode_image(uploaded_file_2)
    response = chain.invoke({"input": question,"image1": image1,"image2": image2})
    st.write(response.content)