import os
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama,OllamaEmbeddings

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#GOOGLE_API_KEY = "AIzaSyALRgNQKeDBZmhuMXBYfdm6y1edHcFFrks"

# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)

# embeddings = OllamaEmbeddings(model="llama3.2")
# llm = ChatOllama(model="llama3.2")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")

examples = [
    {"input": "Lo", "output": """[
 {
  "medicine_name_suggested": "Loratadine,10 mg,Oral",
  "confidence_score": 0.9,
  "medicine_name": "Loratadine",
  "strength": "10 mg",
  "mode_of_administration": "Oral"
 },
 {
  "medicine_name_suggested": "Levocetirizine,5 mg,Oral",
  "confidence_score": 0.5,
  "medicine_name": "Levocetirizine",
  "strength": "5 mg",
  "mode_of_administration": "Oral"
 },
 {
  "medicine_name_suggested": "Olopatadine,665 mcg per spray,Nasal Spray",
  "confidence_score": 0.1,
  "medicine_name": "Olopatadine",
  "strength": "665 mcg per spray",
  "mode_of_administration": "Nasal Spray"
 }
]"""},
    {"input": "Mometasone", "output": """[
  {
    "medicine_name_suggested": "Mometasone Furoate,50 mcg per spray,Nasal Spray",
    "confidence_score": 0.9,
    "medicine_name": "Mometasone Furoate",
    "strength": "50 mcg per spray",
    "mode_of_administration": "Nasal Spray"
  }
]"""},
{"input": "abcdefg", "output": """[
  {
    "medicine_name_suggested": "NoMatchFound"
  }
]"""}
]

# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt_template = ChatPromptTemplate.from_messages(
    [("system", """
               You are a pharmacist who needs to find correct medicine names.
               Use the provided context to look up for medicine names and find the best matched.
               Return a json object as answer containing below:
               one or more row from the context as medicine_name_suggested and the confidence score as confidence_score for every row
                {context}
                """),
     few_shot_prompt,
     ("human", "{input}")
     ]
)

document = TextLoader("medicine.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks,embeddings)
retriever = vector_store.as_retriever()

qa_chain = create_stuff_documents_chain(llm,prompt_template)
rag_chain = create_retrieval_chain(retriever,qa_chain)

question = input("Enter a medicine name")

if question:
    response = rag_chain.invoke({"input": question})
    print(response["answer"])


