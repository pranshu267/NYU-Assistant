import streamlit as st
from PIL import Image
from retriever import get_qa_chain

st.set_page_config(page_title="NYUAssistant", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("src/style.css") 


logo_image = Image.open("NYU_logo.png") 
st.image(logo_image, width=100) 

st.write("# NYUAssistant") 
st.write("Welcome to the NYUAssistant. How can I assist you today?")

question = st.text_input("Please enter your question here:", "")

if st.button("Submit"):

    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
