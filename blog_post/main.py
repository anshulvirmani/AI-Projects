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

role = st.text_input("Enter your role", placeholder="Product Management..")
designation = st.text_input("Enter your designation", placeholder="Head of Products..")
industry = st.text_input("Enter the industry that you work in", placeholder="Retail, edtech etc..")

topic = st.text_input("Enter the topic of your LinkedIn post", placeholder="Create a LinkedIn post about xyz")

openai_api_key = st.text_input(
    "OpenAI API Key",
    type = "password"
)

template = """
You are an expoert content writer for LinkedIn. A customer needs your services to write a LinkedIn post for him. The customer has provided the pdf that you will refer to for creating the content of the pos\
The customer has also provided the details about himself or herself like the designation, role, industry etc. You need to take this information into account and create the post that this person would have created\
Here are the details:
{role}
{designation}
{industry}
{topic}

At the end of the post, mention the references that you took from the uplodaded file. Here is the format:
LinkedIn Post:
Put the post here

References:
List down the references you took from the file to generate the content.
"""

prompt = PromptTemplate(
    input_variables = ["role","designation","industry", "topic"],
    template = template
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

if st.button('Submit'):
    # Check if the user has entered a value
    if openai_api_key.startswith("sk-"):
        if role:
            if designation:
                if industry:
                    if topic:
                        if uploaded_file is not None:
                            question = prompt.format(role=role,designation = designation, industry = industry, topic = topic )
                            response = generate_response(uploaded_file, question)
                            st.write(response)
                        else:
                            st.warning("Upload the file first")
                            st.stop()
                    else:
                        st.warning("Enter the topic of the LinkedIn post")
                        st.stop()
                else:
                    st.warning("Enter the domain of the industry that you work in")
                    st.stop()
            else:
                st.warning("Enter your designation")
                st.stop()
        else:
            st.warning("Enter your role")
            st.stop()
    else:
        st.warning("Enter the key correctly")
        st.stop()



