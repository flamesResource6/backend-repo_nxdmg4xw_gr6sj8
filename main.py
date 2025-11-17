import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, AsyncGenerator
import requests

from database import create_document, get_documents, db
from schemas import ChatRequest, ChatResponse, Conversation, Message

# Optional provider SDKs
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

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


# Expanded catalog across providers
MODEL_REGISTRY: List[ModelInfo] = [
    ModelInfo(id="echo:mini", name="Echo Mini (Demo)", provider="demo", description="Replies with what you say (for testing)."),
    # OpenAI (multimodal-capable models highlighted)
    ModelInfo(id="openai:gpt-4o", name="GPT-4o (Multimodal)", provider="openai"),
    ModelInfo(id="openai:gpt-4o-mini", name="GPT-4o Mini (Multimodal)", provider="openai"),
    ModelInfo(id="openai:gpt-4.1", name="GPT-4.1", provider="openai"),
    # Anthropic
    ModelInfo(id="anthropic:claude-3-5-sonnet-20240620", name="Claude 3.5 Sonnet", provider="anthropic"),
    ModelInfo(id="anthropic:claude-3-opus-20240229", name="Claude 3 Opus", provider="anthropic"),
    ModelInfo(id="anthropic:claude-3-haiku-20240307", name="Claude 3 Haiku", provider="anthropic"),
    # Google Gemini (multimodal)
    ModelInfo(id="google:gemini-1.5-pro", name="Gemini 1.5 Pro (Multimodal)", provider="google"),
    ModelInfo(id="google:gemini-1.5-flash", name="Gemini 1.5 Flash (Multimodal)", provider="google"),
    # Hugging Face popular models
    ModelInfo(id="hf:tiiuae/falcon-7b-instruct", name="Falcon 7B Instruct", provider="huggingface"),
    ModelInfo(id="hf:google/flan-t5-base", name="FLAN-T5 Base", provider="huggingface"),
    ModelInfo(id="hf:facebook/opt-1.3b", name="OPT 1.3B", provider="huggingface"),
    ModelInfo(id="hf:OpenAssistant/oasst-sft-1-pythia-12b", name="OpenAssistant Pythia 12B", provider="huggingface"),
    ModelInfo(id="hf:tiiuae/falcon-7b", name="Falcon 7B", provider="huggingface"),
    ModelInfo(id="hf:bigscience/bloom-1b7", name="BLOOM 1.7B", provider="huggingface"),
    ModelInfo(id="hf:bigscience/bloomz-1b7", name="BLOOMZ 1.7B", provider="huggingface"),
    ModelInfo(id="hf:databricks/dolly-v2-3b", name="Dolly v2 3B", provider="huggingface"),
    ModelInfo(id="hf:meta-llama/Llama-2-7b-chat-hf", name="Llama 2 7B Chat (HF)", provider="huggingface"),
    ModelInfo(id="hf:mistralai/Mistral-7B-Instruct-v0.1", name="Mistral 7B Instruct v0.1", provider="huggingface"),
    ModelInfo(id="hf:mistralai/Mixtral-8x7B-Instruct-v0.1", name="Mixtral 8x7B Instruct", provider="huggingface"),
]


def _provider_status() -> Dict[str, bool]:
    return {
        "demo": True,
        "openai": bool(os.getenv("OPENAI_API_KEY")) and (OpenAI is not None),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")) and (anthropic is not None),
        "google": bool(os.getenv("GOOGLE_API_KEY")) and (genai is not None),
        "huggingface": bool(os.getenv("HUGGINGFACE_API_KEY")),
    }


@app.get("/api/providers")
def providers():
    status = _provider_status()
    # redact values: just booleans
    return {"providers": status}


@app.get("/api/models")
def list_models(q: Optional[str] = None, limit: int = 200, hide_unconfigured: bool = True):
    items = MODEL_REGISTRY

    if hide_unconfigured:
        status = _provider_status()
        items = [m for m in items if status.get(m.provider, False)]

    if q:
        ql = q.lower()
        items = [m for m in items if ql in m.id.lower() or ql in (m.name.lower()) or ql in m.provider.lower()]
    return items[: max(1, min(limit, 500))]


# -------------------- Conversations --------------------
@app.get("/api/conversations")
def get_conversations(limit: int = 50):
    try:
        items = get_documents("conversation", {}, limit)
        for it in items:
            if "_id" in it:
                it["id"] = str(it["_id"])  # mirror to id field for frontend
                it.pop("_id", None)
        return items
    except Exception:
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
        msgs.sort(key=lambda x: x.get("created_at", 0))
        return msgs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Chat Helpers --------------------

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
            "return_full_text": False,
        },
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"HF error: {resp.text[:200]}")
        data = resp.json()
        if isinstance(data, list) and data:
            generated = data[0].get("generated_text") or data[0].get("generated_texts") or ""
            return generated.strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        return str(data)[:500]
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"HF request failed: {str(e)}")


def openai_convert_messages(messages: List[Dict[str, str]]):
    # Convert messages to OpenAI chat format with optional image_url parts
    converted = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        image_url = m.get("image_url")
        if image_url:
            content_parts = [
                {"type": "text", "text": content or ""},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            converted.append({"role": role, "content": content_parts})
        else:
            converted.append({"role": role, "content": content})
    return converted


def call_openai(messages: List[Dict[str, str]], model: str) -> str:
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")
    client = OpenAI(api_key=api_key)
    try:
        oa_messages = openai_convert_messages(messages)
        resp = client.chat.completions.create(model=model, messages=oa_messages, temperature=0.7, max_tokens=500)
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


def call_anthropic(messages: List[Dict[str, str]], model: str) -> str:
    if anthropic is None:
        raise HTTPException(status_code=500, detail="anthropic package not installed")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in environment")
    client = anthropic.Anthropic(api_key=api_key)
    # Map OpenAI-style roles to Anthropic format
    system = None
    content_msgs = []
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content")
        else:
            content_msgs.append({"role": m.get("role"), "content": m.get("content")})
    try:
        resp = client.messages.create(model=model, system=system, messages=content_msgs, max_tokens=500, temperature=0.7)
        # Anthropics returns a list of content blocks
        text = "".join([b.text for b in resp.content if getattr(b, "type", "text") == "text"])  # type: ignore
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anthropic error: {str(e)}")


def call_google(messages: List[Dict[str, str]], model: str) -> str:
    if genai is None:
        raise HTTPException(status_code=500, detail="google-generativeai package not installed")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set in environment")
    genai.configure(api_key=api_key)
    try:
        gmodel = genai.GenerativeModel(model)
        # Convert to the prompt style Gemini expects
        parts = []
        system = None
        for m in messages:
            role = m.get("role")
            img = m.get("image_url")
            txt = m.get("content", "")
            if role == "system":
                system = txt
            else:
                role_map = "user" if role == "user" else "model"
                role_parts = []
                if txt:
                    role_parts.append(txt)
                if img:
                    role_parts.append({"mime_type": "image/jpeg", "file_uri": img})
                parts.append({"role": role_map, "parts": role_parts})
        chat = gmodel.start_chat(history=[])
        if system:
            parts.insert(0, {"role": "user", "parts": [f"System instruction: {system}"]})
        resp = chat.send_message(parts)
        return resp.text or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Gemini error: {str(e)}")


# -------------------- Chat (non-streaming) --------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    model_id = payload.model

    # Ensure conversation exists
    conversation_id = payload.conversation_id
    if not conversation_id:
        first_user = next((m for m in payload.messages if m.role == "user"), None)
        title = (first_user.content[:40] + "...") if first_user else "New Chat"
        conv = Conversation(title=title, model=model_id)
        conversation_id = create_document("conversation", conv)

    # Save incoming messages
    try:
        for m in payload.messages:
            if m.role in ("user", "system"):
                create_document("message", m)
    except Exception:
        pass

    msgs = [m.dict() if hasattr(m, "dict") else (m.model_dump() if hasattr(m, "model_dump") else m) for m in payload.messages]

    reply_text = ""
    if model_id.startswith("echo:"):
        last_user = next((m for m in reversed(payload.messages) if m.role == "user"), None)
        reply_text = f"You said: {last_user.content if last_user else ''}"
    elif model_id.startswith("hf:"):
        hf_model = model_id.split("hf:", 1)[1]
        prompt = build_prompt(msgs)
        reply_text = call_huggingface_generation(hf_model, prompt)
    elif model_id.startswith("openai:"):
        oi_model = model_id.split("openai:", 1)[1]
        reply_text = call_openai(msgs, oi_model)
    elif model_id.startswith("anthropic:"):
        an_model = model_id.split("anthropic:", 1)[1]
        reply_text = call_anthropic(msgs, an_model)
    elif model_id.startswith("google:"):
        go_model = model_id.split("google:", 1)[1]
        reply_text = call_google(msgs, go_model)
    else:
        raise HTTPException(status_code=400, detail="Unknown model provider")

    try:
        reply_message = Message(conversation_id=conversation_id, role="assistant", content=reply_text)
        create_document("message", reply_message)
    except Exception:
        pass

    return ChatResponse(conversation_id=conversation_id, reply=reply_text, model=model_id)


# -------------------- Chat (streaming SSE) --------------------
@app.post("/api/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest):
    model_id = payload.model

    # Ensure conversation exists
    conversation_id = payload.conversation_id
    if not conversation_id:
        first_user = next((m for m in payload.messages if m.role == "user"), None)
        title = (first_user.content[:40] + "...") if first_user else "New Chat"
        conv = Conversation(title=title, model=model_id)
        conversation_id = create_document("conversation", conv)

    # Persist user/system messages
    try:
        for m in payload.messages:
            if m.role in ("user", "system"):
                create_document("message", m)
    except Exception:
        pass

    msgs = [m.dict() if hasattr(m, "dict") else (m.model_dump() if hasattr(m, "model_dump") else m) for m in payload.messages]

    async def provider_stream() -> AsyncGenerator[str, None]:
        # Send meta
        yield f"event: meta\ndata: {{\"conversation_id\": \"{conversation_id}\", \"model\": \"{model_id}\"}}\n\n"

        # Providers
        text = ""
        try:
            if model_id.startswith("echo:"):
                last_user = next((m for m in reversed(payload.messages) if m.role == "user"), None)
                text = f"You said: {last_user.content if last_user else ''}"
                for ch in text:
                    yield f"event: token\ndata: {ch}\n\n"
            elif model_id.startswith("hf:"):
                prompt = build_prompt(msgs)
                hf_model = model_id.split("hf:", 1)[1]
                text = call_huggingface_generation(hf_model, prompt)
                chunk = 32
                for i in range(0, len(text), chunk):
                    yield f"event: token\ndata: {text[i:i+chunk]}\n\n"
            elif model_id.startswith("openai:"):
                oi_model = model_id.split("openai:", 1)[1]
                if OpenAI is None:
                    raise HTTPException(status_code=500, detail="openai package not installed")
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")
                client = OpenAI(api_key=api_key)
                try:
                    oa_messages = openai_convert_messages(msgs)
                    stream = client.chat.completions.create(model=oi_model, messages=oa_messages, temperature=0.7, max_tokens=500, stream=True)
                    collected = []
                    for event in stream:  # type: ignore
                        delta = event.choices[0].delta.content or ""
                        if delta:
                            collected.append(delta)
                            yield f"event: token\ndata: {delta}\n\n"
                    text = "".join(collected)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"OpenAI stream error: {str(e)}")
            elif model_id.startswith("anthropic:"):
                an_model = model_id.split("anthropic:", 1)[1]
                if anthropic is None:
                    raise HTTPException(status_code=500, detail="anthropic package not installed")
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in environment")
                client = anthropic.Anthropic(api_key=api_key)
                system = None
                content_msgs = []
                for m in msgs:
                    if m.get("role") == "system":
                        system = m.get("content")
                    else:
                        content_msgs.append({"role": m.get("role"), "content": m.get("content")})
                with client.messages.stream(model=an_model, system=system, messages=content_msgs, max_tokens=500, temperature=0.7) as stream:  # type: ignore
                    collected = []
                    for event in stream:
                        if event.type == "content_block_delta":
                            delta = event.delta.get("text", "")
                            if delta:
                                collected.append(delta)
                                yield f"event: token\ndata: {delta}\n\n"
                    text = "".join(collected)
            elif model_id.startswith("google:"):
                go_model = model_id.split("google:", 1)[1]
                if genai is None:
                    raise HTTPException(status_code=500, detail="google-generativeai package not installed")
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set in environment")
                genai.configure(api_key=api_key)
                gmodel = genai.GenerativeModel(go_model)
                system = None
                parts = []
                for m in msgs:
                    role = m.get("role")
                    if role == "system":
                        system = m.get("content")
                    else:
                        img = m.get("image_url")
                        txt = m.get("content", "")
                        role_parts = []
                        if txt:
                            role_parts.append(txt)
                        if img:
                            role_parts.append({"mime_type": "image/jpeg", "file_uri": img})
                        parts.append({"role": "user" if role == "user" else "model", "parts": role_parts})
                if system:
                    parts.insert(0, {"role": "user", "parts": [f"System instruction: {system}"]})
                # Gemini doesn't expose SSE directly via SDK in python; stream via generate_content with stream=True
                try:
                    stream = gmodel.generate_content(parts, stream=True)
                    collected = []
                    for chunk in stream:
                        if hasattr(chunk, "text") and chunk.text:
                            collected.append(chunk.text)
                            yield f"event: token\ndata: {chunk.text}\n\n"
                    text = "".join(collected)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Google stream error: {str(e)}")
            else:
                text = "Unknown model provider"
                yield f"event: token\ndata: {text}\n\n"
        except HTTPException as e:
            err = f"Error: {e.detail}"
            for ch in err:
                yield f"event: token\ndata: {ch}\n\n"
            text = err
        except Exception as e:
            err = f"Error: {str(e)}"
            for ch in err:
                yield f"event: token\ndata: {ch}\n\n"
            text = err

        # done event
        yield f"event: done\ndata: {{\"conversation_id\": \"{conversation_id}\"}}\n\n"

        # Persist full text after completion
        try:
            reply_message = Message(conversation_id=conversation_id, role="assistant", content=text)
            create_document("message", reply_message)
        except Exception:
            pass

    return StreamingResponse(provider_stream(), media_type="text/event-stream")


# -------------------- Infra Test --------------------
@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
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
