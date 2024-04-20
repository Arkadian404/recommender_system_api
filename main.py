from http import HTTPStatus
from typing import List

from pydantic import BaseModel
import uvicorn
import KNN
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

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


connect_string = "mysql+pymysql://root:123456@127.0.0.1:3306/filtro_jwt"
recommendations_service = KNN.KNN(connect_string)


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
