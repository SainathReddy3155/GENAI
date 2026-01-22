from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
import streamlit as st
import os
load_dotenv()
st.title("Chatbot using ChatOPENAI")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7,max_completion_tokens=100)
memory=ConversationBufferMemory()
conversation=ConversationChain(
    memory=memory,
    llm=llm,
    verbose=False
)
user_input=st.text_input("Enter Your query : ")
if user_input:
    # if user_input=='Q' or user_input.lower()=='quit':
    #     print("Thanks for using the bot")
    result=conversation.invoke(user_input)
    final_result=result['response']
    print("final_result : ",final_result)
    st.write(final_result)
