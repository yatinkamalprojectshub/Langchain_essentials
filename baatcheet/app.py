import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.llms import Ollama

# ------------------------------
# Load env keys
# ------------------------------
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# ------------------------------
# Streamlit page setup
# ------------------------------
st.set_page_config(page_title="Open-Source Chat Assistant", layout="wide")
st.title("🤖 YATIN_AI")

# ------------------------------
# Models
# ------------------------------
groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
ollama_llm = Ollama(model="mistral")

# ------------------------------
# Session state for chat history
# ------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely."}
    ]

# ------------------------------
# Display chat history
# ------------------------------
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

# ------------------------------
# Chat input (like ChatGPT)
# ------------------------------
if prompt := st.chat_input("Kuch bhi pooch lo..."):
    # Add user input
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Sochne do..."):
            try:
                start = time.process_time()
                # Default: Use Groq (fast)
                response = groq_llm.invoke(prompt)
                answer = response.content
                elapsed = time.process_time() - start
                st.write(answer)
                st.caption(f"⏱️ Response time: {elapsed:.2f}s (via Groq)")
            except Exception:
                # Fallback to Ollama if Groq fails
                answer = ollama_llm.invoke(prompt).content
                st.write(answer)
                st.caption("⚡ Answered locally via Ollama")

    # Save assistant response
    st.session_state["messages"].append({"role": "assistant", "content": answer})

# ------------------------------
# Optional: Clear chat button
# ------------------------------

if st.button("🗑️ Clear Chat"):
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely."}
    ]
    # Force Streamlit to rerun by using st.experimental_rerun safely
    import streamlit as st_lib
    if hasattr(st_lib, "experimental_rerun"):
        st_lib.experimental_rerun()
    else:
        # fallback: do nothing, session state reset is enough
        st.success("Chat cleared!")

