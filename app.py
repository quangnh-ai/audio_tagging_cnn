from turtle import st
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import uvicorn
import h5py

from utils.retrieval_model import Retrieval
from utils.retrieval_model import get_arg

app = FastAPI()

class Request(BaseModel):
    q: str

@app.post("/retrieval")
async def retrieval(req: Request):
    req = req.dict()
    query = req["query"]
    return {"result": retrieval_model.retrieval(query)}

@app.get("/audio_tag/{tag}", status_code=200)
async def text_query(tag: str, q: str):
    txt_query = q
    if txt_query == "" or txt_query == None or txt_query == " ":
        return {
            "message": "query string is null"
        }
    return retrieval_model.retrieval(txt_query)

if __name__ == "__main__":
    retrieval_model = Retrieval(get_arg())
    uvicorn.run(app, host="0.0.0.0", port=5432)
