from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent, run_agent_stream
from extract import extract_text_from_pdf
from chunk_embed import load_extracted_text, store_in_chromadb
from dotenv import load_dotenv
from typing import Optional
import uvicorn
import shutil
import uuid
import os
from datetime import datetime
from collections import defaultdict
from prompts import SUMMARY_PROMPT

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# per tab sessions
sessions = {}

# visitor tracking
visitor_stats = {
    "total_requests": 0,
    "unique_visitors": set(),
    "requests_by_path": defaultdict(int),
    "start_time": datetime.now().isoformat(),
    "last_request": None
}


@app.middleware("http")
async def track_visitors(request: Request, call_next):
    """Track visitor requests and unique IPs"""
    # get client IP (handles proxies like Railway)
    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()
    
    # count request
    visitor_stats["total_requests"] += 1
    visitor_stats["unique_visitors"].add(client_ip)
    visitor_stats["requests_by_path"][request.url.path] += 1
    visitor_stats["last_request"] = datetime.now().isoformat()
    
    response = await call_next(request)
    return response


class Question(BaseModel):
    question: str
    tab_id:   Optional[str] = None


class ClearRequest(BaseModel):
    tab_id: Optional[str] = None


@app.get("/stats")
async def get_stats():
    """Return visitor analytics (public endpoint)"""
    return {
        "total_requests": visitor_stats["total_requests"],
        "unique_visitors": len(visitor_stats["unique_visitors"]),
        "requests_by_path": dict(visitor_stats["requests_by_path"]),
        "uptime_since": visitor_stats["start_time"],
        "last_request": visitor_stats["last_request"],
        "active_sessions": len(sessions)
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_pdf(
    file:   UploadFile = File(...),
    tab_id: Optional[str] = Form(None)
):
    session_key = tab_id or str(uuid.uuid4())

    try:
        # use a project-local tmp directory (works cross-platform)
        base_tmp = os.path.join(os.getcwd(), "tmp")
        data_dir = os.path.join(base_tmp, "data")
        extracted_base = os.path.join(base_tmp, "extracted")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(extracted_base, exist_ok=True)

        pdf_path = os.path.join(data_dir, f"{session_key}_{file.filename}")
        extracted_dir = os.path.join(extracted_base, session_key)

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        collection_name = f"pdf_{uuid.uuid4().hex[:8]}"

        print(f"Processing: {file.filename} — session {session_key[:8]}")

        extract_text_from_pdf(pdf_path, output_dir=extracted_dir)
        texts       = load_extracted_text(extracted_dir)
        chunk_count = store_in_chromadb(texts, collection_name)

        sessions[session_key] = {
            "collection_name": collection_name,
            "pdf_name":        file.filename,
            "ready":           True,
            "history":         []
        }

        return JSONResponse({
            "status":   "ready",
            "filename": file.filename,
            "chunks":   chunk_count,
            "tab_id":   session_key
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/ask")
async def ask(q: Question, request: Request):
    session_key = q.tab_id
    # detect streaming request header
    stream = request.headers.get("x-stream", "false").lower() == "true"

    # if no PDF uploaded — still allow web search
    if not session_key or session_key not in sessions:
        # no PDF uploaded — allow web search
        if stream:
            def event_stream():
                for ev in run_agent_stream(q.question, None, []):
                    yield f"data: {json.dumps(ev)}\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        try:
            answer, tool_used = run_agent(
                q.question,
                None,        # no collection
                []           # empty history
            )
            return {"answer": answer, "tool_used": tool_used, "status": "ok"}
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "tool_used": "", "status": "error"}

    # PDF uploaded — use session
    user_session = sessions[session_key]
    # PDF uploaded — use session
    user_session = sessions[session_key]
    if stream:
        def event_stream():
            for ev in run_agent_stream(q.question, user_session["collection_name"], user_session["history"]):
                yield f"data: {json.dumps(ev)}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    try:
        answer, tool_used = run_agent(
            q.question,
            user_session["collection_name"],
            user_session["history"]
        )
        return {"answer": answer, "tool_used": tool_used, "status": "ok"}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "tool_used": "", "status": "error"}

@app.post("/clear")
async def clear(req: ClearRequest):
    if req.tab_id and req.tab_id in sessions:
        sessions[req.tab_id]["history"] = []
    return {"status": "cleared"}


@app.get("/status")
async def status():
    return {"active_sessions": len(sessions)}


def is_summary_request(question: str) -> bool:
    keywords = ["summarize", "summary", "overview", "what is this about", "main topics", "key points"]
    return any(kw in question.lower() for kw in keywords)


if __name__ == "__main__":
    # Use reload only in development mode
    reload = os.getenv("ENVIRONMENT") != "production"
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=reload)