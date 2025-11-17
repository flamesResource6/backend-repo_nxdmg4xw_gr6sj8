import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests

from database import create_document, get_documents, db
from schemas import ChatRequest, ChatResponse, Conversation, Message

app = FastAPI(title="Multi-Model Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


# -------------------- Model Registry --------------------
class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    description: Optional[str] = None
    context_length: Optional[int] = None


# Minimal starter set; you can expand this list over time
MODEL_REGISTRY: List[ModelInfo] = [
    ModelInfo(id="echo:mini", name="Echo Mini (Demo)", provider="demo", description="Replies with what you say (for testing)."),
    ModelInfo(id="hf:tiiuae/falcon-7b-instruct", name="Falcon 7B Instruct", provider="huggingface"),
    ModelInfo(id="hf:google/flan-t5-base", name="FLAN-T5 Base", provider="huggingface"),
    ModelInfo(id="hf:facebook/opt-1.3b", name="OPT 1.3B", provider="huggingface"),
    ModelInfo(id="hf:OpenAssistant/oasst-sft-1-pythia-12b", name="OpenAssistant Pythia 12B", provider="huggingface"),
]


@app.get("/api/models", response_model=List[ModelInfo])
def list_models():
    return MODEL_REGISTRY


# -------------------- Conversations --------------------
@app.get("/api/conversations")
def get_conversations(limit: int = 50):
    try:
        items = get_documents("conversation", {}, limit)
        # Convert ObjectId to string if present
        for it in items:
            if "_id" in it:
                it["id"] = str(it["_id"])  # mirror to id field for frontend
                it.pop("_id", None)
        return items
    except Exception as e:
        return []


class CreateConversationRequest(BaseModel):
    title: str
    model: str


@app.post("/api/conversations")
def create_conversation(payload: CreateConversationRequest):
    try:
        conv = Conversation(title=payload.title, model=payload.model)
        conv_id = create_document("conversation", conv)
        return {"id": conv_id, "title": conv.title, "model": conv.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str, limit: int = 200):
    try:
        msgs = get_documents("message", {"conversation_id": conversation_id}, limit)
        for m in msgs:
            if "_id" in m:
                m["id"] = str(m["_id"])  # mirror
                m.pop("_id", None)
        # Sort by created time if available
        msgs.sort(key=lambda x: x.get("created_at", 0))
        return msgs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Chat Endpoint --------------------

def build_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def call_huggingface_generation(model_id: str, prompt: str) -> str:
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY not set in environment")

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False
        }
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"HF error: {resp.text[:200]}")
        data = resp.json()
        # HF returns list of dicts for text-generation
        if isinstance(data, list) and data:
            generated = data[0].get("generated_text") or data[0].get("generated_texts") or ""
            return generated.strip()
        # Some models return dict with 'generated_text'
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        # Fallback
        return str(data)[:500]
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"HF request failed: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    # Determine model
    model_id = payload.model

    # Ensure conversation exists
    conversation_id = payload.conversation_id
    if not conversation_id:
        # Create new conversation with title from first user message
        first_user = next((m for m in payload.messages if m.role == "user"), None)
        title = (first_user.content[:40] + "...") if first_user else "New Chat"
        conv = Conversation(title=title, model=model_id)
        conversation_id = create_document("conversation", conv)

    # Save incoming messages (only those without id)
    try:
        for m in payload.messages:
            # Persist user/assistant/system messages except assistant for the latest since we'll generate a new reply
            if m.role in ("user", "system"):
                create_document("message", m)
    except Exception:
        # If DB not available, continue without persistence
        pass

    # Build prompt and call provider
    # Convert messages to dicts for prompt
    msgs = [m.dict() if hasattr(m, "dict") else (m.model_dump() if hasattr(m, "model_dump") else m) for m in payload.messages]
    prompt = build_prompt(msgs)

    reply_text = ""
    if model_id.startswith("echo:"):
        # Simple echo behavior for demo
        last_user = next((m for m in reversed(payload.messages) if m.role == "user"), None)
        reply_text = f"You said: {last_user.content if last_user else ''}"
    elif model_id.startswith("hf:"):
        hf_model = model_id.split("hf:", 1)[1]
        reply_text = call_huggingface_generation(hf_model, prompt)
    else:
        raise HTTPException(status_code=400, detail="Unknown model provider")

    # Save assistant reply
    try:
        reply_message = Message(conversation_id=conversation_id, role="assistant", content=reply_text)
        create_document("message", reply_message)
    except Exception:
        pass

    return ChatResponse(conversation_id=conversation_id, reply=reply_text, model=model_id)


# -------------------- Infra Test --------------------
@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
