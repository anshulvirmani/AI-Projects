import streamlit as st
from langchain_openai import OpenAI
from langchain import PromptTemplate

st.set_page_config(
    page_title = "Extract key information from Product Reviews"
)

st.title("Extract key information from Product Reviews")

#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.write("""
    Extract key information from a product review.
             
    - Sentiment
    - How long did it take to deliver?
    - How was the price perceived?
    """)

with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")

st.markdown("## Enter your OpenAI API Key")

openai_api_key = st.text_input(
    "OpenAI API Key",
    type = "password"
)

st.markdown("## Enter the product review")

review = st.text_area("Enter your product review:", placeholder="Type your review here...", height=300, label_visibility="collapsed")

def generate_response(review):
    llm = OpenAI(openai_api_key=openai_api_key)
    template = """
    I will give you a product review and you have to extract information from the review. There are three things that you have to extract:
    Sentiment: this tells if the review was positive, negative or neutral
    Delivery time: this tells how long it took to deliver the product. this should be in the number of days. if there is no information of the delivery time then mention 'no information'
    Price perception: this tells how was the price of the product perceived by the user. 4 possible values: cheap, normal, expensive, no information

    Here are some examples:
    Review: The product was very good. The delivery was slow, it took 3 days for the product to reach me. However, i am very excited to get this product at such an amazing price point. Love it!
    Sentiment: Positive
    Delivery time: 3 days
    Price Perception: cheap

    Here is the review posted by the user:
    {review}
    """
    prompt = PromptTemplate(
        input_variables = ["review"],
        template = template
    )
    query = prompt.format(review=review)
    response = llm(query, max_tokens=2048)
    return st.write(response)

if review:
    if not openai_api_key.startswith("sk-"):
        st.warning("Enter OpenAI API Key")
        st.stop()
    if openai_api_key.startswith("sk-"):
        generate_response(review)


