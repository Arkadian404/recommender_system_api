import os
from dotenv import load_dotenv
from typing import List
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import KNN
import Chatbot
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
app = FastAPI()
origins = ["*"]

DB_USER = "root"
DB_PASSWORD = "123456"
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = 3306
DATABASE = "filtro_jwt"


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class Response(BaseModel):
    recommendations: List[int] | None
    detail: str | None

connect_string = ('mysql+pymysql://{}:{}@{}:{}/{}'
                  .format(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DATABASE))
recommendations_service = KNN.KNN(connect_string)
user_memory_dicts = {}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/recommendations/{user_id}", response_model=Response)
async def recommendations(user_id: int):
    re = recommendations_service.get_recommendations(user_id)
    return Response(recommendations=re, detail="Success")


@app.get("/chatbot/invoke/{user_id}")
async def agent_invoke(user_id: str, user_query: str):
    if user_id not in user_memory_dicts:
        user_memory_dicts[user_id] = ConversationBufferMemory(memory_key='history',
                                                              input_key='input',
                                                              output_key='output',
                                                              return_messages=True)
    memory = user_memory_dicts[user_id]
    chatbot = Chatbot.SQLAgent(connect_string, memory)
    response = chatbot.run(user_query)
    return response["output"]