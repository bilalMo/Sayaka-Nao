from dotenv import load_dotenv
import os
load_dotenv()
api_key_env_name="google_api_key"
api_key = os.getenv(api_key_env_name)

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load document
loader = TextLoader('test.txt', encoding="utf8")
document = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(document)

# Embedding and vectorstore
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = FAISS.from_documents(text_chunks, embeddings)
retriever = vectorstore.as_retriever()

# Prompt template
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM setup
llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=None)
output_parser = StrOutputParser()

# RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

# Invoke with input question
with open('tanya.txt', 'r', encoding='utf-8') as file:
    text_content = file.read()

print(rag_chain.invoke(text_content))
