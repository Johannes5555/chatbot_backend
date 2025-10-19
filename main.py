import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    client = None

app = FastAPI(title="AI Chatbot Demo API", version="0.1.0")

# CORS: allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str = Field(..., description="system|user|assistant")
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = Field(default="gpt-4o-mini", description="OpenAI model id")
    temperature: Optional[float] = 0.2

class ChatResponse(BaseModel):
    reply: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check dependencies.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY env variable")
    # Ensure at least one user message exists
    if not any(m.role == "user" for m in req.messages):
        raise HTTPException(status_code=422, detail="At least one user message required")
    try:
        completion = client.chat.completions.create(
            model=req.model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            temperature=req.temperature,
        )
        reply = completion.choices[0].message.content
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))