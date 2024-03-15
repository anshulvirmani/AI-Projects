import streamlit as st
import fitz  # PyMuPDF
from langchain_openai import OpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader

st.title('QnA from a long pdf document')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

question = st.text_input("Enter your question", placeholder="What is the user talking about")

openai_api_key = st.text_input(
    "OpenAI API Key",
    type = "password"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)

def generate_response(text, question):
    llm = OpenAI(openai_api_key=openai_api_key)
    reader = PdfReader(text)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())

    docs = text_splitter.create_documents(formatted_document)
    embeddings = OpenAIEmbeddings()
    stored_embeddings = FAISS.from_documents(docs, embeddings)

    QA_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=stored_embeddings.as_retriever()
    )

    return QA_chain.run(question)


if question:
    if not openai_api_key.startswith("sk-"):
        st.warning("Enter OpenAI API Key")
        st.stop()
    if openai_api_key.startswith("sk-"):
        if uploaded_file is not None:
            response = generate_response(uploaded_file, question)
            st.write(response)


