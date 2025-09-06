from fastapi import FastAPI 
from langchain.prompts import ChatPromptTemplate 
#from langchain.chat_models import ChatOpenAI
from langserve import add_routes 
import uvicorn 
import os
from langchain_community.llms import Ollama 
from dotenv import load_dotenv 

load_dotenv() 


#os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
app = FastAPI(
    title = "Langchain Server",
    version="1.0",
    description="A simple API server"
)
''' #use this if we have money
add_routes (
    app,
    ChatOpenAI(),
    path = "/openai"
)
model = ChatOpenAI()
'''
## ollama mistral
llm = Ollama(model="mistral")

prompt1 =ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 =ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")

""" # use these if we have OpenAI tokens
add_routes(
    app,
    prompt1|model, 
    path="/essay"

)
add_routes(
    app,
    prompt2|model, 
    path="/poem"

)
"""
# Add routes for the APIs
add_routes(app, prompt1 | llm, path="/essay")
add_routes(app, prompt2 | llm, path="/poem")

# we basically have created 2 apis in another words
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port= 8000)