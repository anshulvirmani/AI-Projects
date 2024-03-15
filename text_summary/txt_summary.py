import streamlit as st
from langchain_openai import OpenAI
from langchain import PromptTemplate


#Page title and header
st.set_page_config(page_title="Writing Text Summarization")
st.header("Writing Text Summarization")

user_text = st.text_area("Enter your text:", placeholder="Type your text here...", height=300, label_visibility="collapsed")

st.markdown("## Enter your OpenAI API KEY")

openai_api_key = st.text_input(
    "OpenAI API Key",
    type = "password"
)

def generate_response(topic):
    llm = OpenAI(openai_api_key=openai_api_key)
    template = """
    Please summarize this:
    {topic}
    
    Your response should be in this format:
    First, summarize the topic.
    Then, sum the total number of words on it and print the result like this: This summary has X words.
    """
    prompt = PromptTemplate(
        input_variables = ["topic"],
        template = template
    )
    query = prompt.format(topic=topic)
    response = llm(query, max_tokens=2048)
    return st.write(response)

if st.button('Submit'):
    # Check if the user has entered a value
    if not openai_api_key.startswith("sk-"):
        st.warning("Enter OpenAI API Key")
        st.stop()
    if openai_api_key.startswith("sk-"):
        if user_text:
            generate_response(user_text)
        else:
            st.warning("Enter text that you want to summarize")


