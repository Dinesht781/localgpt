from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

LLM_URL = "http://model-runner.docker.internal/engines/llama.cpp/v1/chat/completions"

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(payload: ChatRequest):
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    llm_payload = {
        "model": "ai/smollm2:latest",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            LLM_URL,
            json=llm_payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()

    except requests.HTTPError:
        raise HTTPException(
            status_code=503,
            detail="Model is loading or unavailable. Please try again shortly."
        )
