import streamlit as st
from langchain_openai import OpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

#Page title and header
st.set_page_config(page_title="AI text summarizer")
st.header("AI text summarizer")

#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("ChatGPT can not summarize long texts. Now you can do it with this App")

with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")

st.markdown("## Enter your OpenAI API KEY")

openai_api_key = st.text_input(
    "OpenAI API Key",
    type = "password"
)

st.markdown("## Upload the text file you want to summarize")

# Add a file uploader widget to your Streamlit app
article = st.file_uploader("Choose a TXT file", type="txt")

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], 
    chunk_size=5000,
    chunk_overlap=350
)

article_chunks = []

def generate_response(article_chunks):
    llm = OpenAI(openai_api_key=openai_api_key)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce"
    )
    article_summary = chain.run(article_chunks)
    st.write(article_summary)

if article is not None:
        if not openai_api_key.startswith("sk-"):
            st.warning("Enter OpenAI API Key")
        if openai_api_key.startswith("sk-"):
            # Read and decode the content of the file
            article_content = article.getvalue().decode("utf-8")
            # Process the content to create chunks
            article_chunks = text_splitter.create_documents([article_content])
            generate_response(article_chunks)