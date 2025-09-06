from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
import streamlit as st
import os 
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


#prompttemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's questions."),
        ("user", "Question: {question}")
    ]
)


#streamlit framework
st.title('Kuch b pooch lo')
input_text = st.text_input("search karke toh dekho")
st.set_page_config(page_title="Kuch B Pooch Lo", layout="centered")


#OpenAI LLM option but using MISTRAL due to my low config device
#llm=ChatOpenAI(model="gpt-3.5-turbo")
llm=ChatOllama(model="mistral")
output_parser =StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
