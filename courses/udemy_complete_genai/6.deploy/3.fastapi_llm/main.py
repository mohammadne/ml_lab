import os
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from ollama import Client
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

client = Client(host=OLLAMA_HOST)

app = FastAPI(title="Ollama FastAPI Tutorial")

# schemas
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    model: str = "gemma2:2b"
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    model: str


# API Route
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = client.chat(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            stream=False
        )
        return ChatResponse(
            response=result["message"]["content"],
            model=result.get("model", request.model),
        )
    except Exception as e:
        return ChatResponse(response=f"[error] {e}", model=request.model)
