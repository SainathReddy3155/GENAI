from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from load_dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def rag_function(pdf_path, question):
  
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

    #Initializing Chroma VD
    vector_db = Chroma.from_documents(chunks,OpenAIEmbeddings(),persist_directory='./chromadb')
    vector_db.persist()

    retriever = vector_db.as_retriever()

    prompt = PromptTemplate.from_template(
        """Answer ONLY using the provided context.
           If the answer is not found, say "Not available in the document."

        Context:
        {context}

        Question: {question}
        """
    )

    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.4,max_completion_tokens=100)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain.invoke(question)


response = rag_function(
    pdf_path="Sainath_Resume_01122025.pdf",
    question="how can i contact sainath?"
)

print(response.content)
