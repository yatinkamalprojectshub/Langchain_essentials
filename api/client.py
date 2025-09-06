""" #if i had money
def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
    json = {'input': {'topic': input_text}})

    return response.json()['output']['content']

def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
    json = {'input': {'topic': input_text}})

    return response.json()['output']['content']
"""
import requests 
import streamlit as st 

def get_ollama_response(input_text, endpoint="poem"):
    url = f"http://localhost:8000/{endpoint}/invoke"
    response = requests.post(
        url,
        json={'input': {'topic': input_text}}
    )

    if response.status_code == 200:
        return response.json()['output']  # output is already a string
    else:
        return f"Error:{response.status_code}-{response.text}"


#streamlit framework
st.title('Kuch b pooch lo')
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    st.write(get_ollama_response(input_text, endpoint="essay"))

if input_text1:
    st.write(get_ollama_response(input_text1, endpoint="poem"))


