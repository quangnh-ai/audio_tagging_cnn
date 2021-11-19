from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import uvicorn
import h5py

from utils.retrieval_model import Retrieval
from utils.retrieval_model import get_arg

app = FastAPI()

class Request(BaseModel):
    query: str

@app.post("/retrieval")
async def retrieval(req: Request):
    req = req.dict()
    query = req["query"]
    return {"result": retrieval_model.retrieval(query)}

@app.get("/text_query/{txt_query}", status_code=200)
async def text_query(txt_query: Optional[str]=None):
    return {"result": retrieval_model.retrieval(txt_query)}

if __name__ == "__main__":
    retrieval_model = Retrieval(get_arg())
    uvicorn.run(app, host="0.0.0.0", port=5432)
