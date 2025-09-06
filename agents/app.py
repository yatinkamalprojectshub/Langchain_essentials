import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Ollama
from langchain.tools.retriever import create_retriever_tool

# ------------------------------
# Setup tools
# ------------------------------
api_wrapper = WikipediaAPIWrapper(top_k_results=1, docs_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

loader = WebBaseLoader("https://docs.smith.langchain.com")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectordb = FAISS.from_documents(documents, embeddings)
retriever = vectordb.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith docs"
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, docs_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools = [
    Tool(name="arxiv", func=arxiv.run, description="Research papers (academic queries)"),
    Tool(name="wiki", func=wiki.run, description="Wikipedia queries (general topics)"),
    Tool(name="retriever", func=retriever_tool.run, description="LangSmith documentation"),
]

# Main LLM
llm = Ollama(model="mistral", temperature=0)

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,                # turn off spam logs
    handle_parsing_errors=True,
    max_iterations=2              # keep fast
)

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Assistant (with Tools + LLM Fallback)")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input box
if prompt := st.chat_input("Ask me anything..."):
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Try agent (tools + llm)
                response = agent.run(prompt)
                if not response or response.strip() == "":
                    # if agent didn't return, fallback
                    response = llm(prompt)
            except Exception:
                # fallback if tool agent fails
                response = llm(prompt)
            st.write(response)

    # Save assistant message
    st.session_state["messages"].append({"role": "assistant", "content": response})
