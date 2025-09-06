#OLLAMA
import streamlit as st
import os
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

load_dotenv()

# Load the GROQ API KEY
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatGroq with Llama3 demo")
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="llama-3.1-8b-instant")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions:{input}
"""
)

def vector_embeddings():
    if "vectors" not in st.session_state:
        # embeddings from mxbai
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # data ingestion
        st.session_state.documents = st.session_state.loader.load()  # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents[:20])  # splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # vector DB

prompt1 = st.text_input("Enter your Question from the Documents")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector store DB is ready")

if "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    if prompt1:
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("response time:", time.process_time() - start)
        st.write(response['answer'])

        with st.expander("References from where its taken"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---------------------------")
