from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()


prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are help assistant. You need to answer the questions asked by the users"),
        ("user","Question :{question}")
    ]
)

#Intializing streamlit
st.title("Chatbot using Ollama")
input_text=st.text_input("Enter your queries :")


#intitalizing LLM
llm=Ollama(model='gemma3:1b')
output_parser=StrOutputParser()
chain=prompt_template|llm|output_parser

if input_text:
    output_area = st.empty()
    streamed_text = ""

    for chunk in chain.stream({"question": input_text}):
        streamed_text += chunk
        output_area.markdown(streamed_text)


