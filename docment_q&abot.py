import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)


st.set_page_config(page_title=" Document Q&A", layout="centered")
st.title("Document Q&A Chatbot")
st.write("Upload a document of any format (PDF, TXT, DOCX, etc.) and ask questions about it!")

# File upload
uploaded_file = st.file_uploader("📎 Upload your document", type=["pdf", "txt", "docx"])

# function to read file and check which format
def read_file(file):
    temp_path = f"uploaded_file.{file.name.split('.')[-1]}"
    print("temp_path : ",temp_path)
    with open(temp_path, "wb") as f:
        f.write(file.read())

    ext = os.path.splitext(temp_path)[1].lower()
    print("ext : ",ext)
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            text = "\n".join([d.page_content for d in docs])
        elif ext == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            text = "\n".join([d.page_content for d in docs])
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_path)
            docs = loader.load()
            text = "\n".join([d.page_content for d in docs])
        else:
            st.error("Unsupported file format.")
            return None
        return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

#ask questions
if uploaded_file is not None:
    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    full_text = read_file(uploaded_file)

    if full_text:
        st.info("Document loaded and ready. Ask your question below!")

        # Define the prompt
        template = """
        You are an AI assistant. Use the document content below to answer the user's question.
        If the answer cannot be found in the document, say so.

        DOCUMENT:
        {document}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = PromptTemplate(template=template, input_variables=["document", "question"])
        chain = LLMChain(prompt=prompt, llm=llm)

        question = st.text_input("💬 Ask a question about your document:")
        if question:
            with st.spinner("Thinking... 🤔"):
                response = chain.invoke({"document": full_text, "question": question})
                answer = response["text"].strip()

            st.subheader("Answer")
            st.text_area("Response", value=answer, height=250)
