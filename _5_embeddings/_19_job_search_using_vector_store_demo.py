import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)
chunks = text_splitter.split_documents(document)
db = Chroma.from_documents(chunks,llm)

# Another way for fetching data from vector store
retriever = db.as_retriever()

text = input("Enter a text.")
embedding_vector = llm.embed_query(text)

docs = db.similarity_search_by_vector(embedding_vector)
docs_alt = retriever.invoke(text)

for doc in docs:
    print(doc.page_content)

print("--------------")
for doc in docs_alt:
    print(doc.page_content)