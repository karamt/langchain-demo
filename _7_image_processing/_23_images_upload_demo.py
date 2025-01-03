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
        ("system", "You are a helpful assistant that can describe images."),
        (
            "human",
            [
                {"type": "text", "text": "{input}"},
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
st.title("Image Analyzer")

uploaded_file = st.file_uploader("Upload your file", type=["jpg", 'png'])

question = st.text_input("Ask a question about image")

if question:
    image = encode_image(uploaded_file)
    response = chain.invoke({"input": question,"image": image})
    st.write(response.content)