from fastapi import FastAPI, UploadFile, File, Response, Cookie
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from agent import run_agent
from extract import extract_text_from_pdf
from chunk_embed import load_extracted_text, store_in_chromadb
from dotenv import load_dotenv
from typing import Optional
import uvicorn
import shutil
import uuid
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

load_dotenv()

app = FastAPI()

# per user sessions — key is session_id cookie
sessions = {}  # session_id → {collection_name, pdf_name, ready, history}


class Question(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def home(response: Response, session_id: Optional[str] = Cookie(None)):
    if not session_id:
        session_id = str(uuid.uuid4())

    with open("templates/index.html", "r", encoding="utf-8") as f:
        content = f.read()

    resp = HTMLResponse(content=content)
    resp.set_cookie("session_id", session_id, samesite="lax")
    return resp


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = Cookie(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        os.makedirs("/tmp/data", exist_ok=True)
        os.makedirs("/tmp/extracted", exist_ok=True)

        # unique path per session — no conflicts between users
        pdf_path      = f"/tmp/data/{session_id}_{file.filename}"
        extracted_dir = f"/tmp/extracted/{session_id}"

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        collection_name = f"pdf_{uuid.uuid4().hex[:8]}"

        print(f"Processing: {file.filename} — session {session_id[:8]}")

        extract_text_from_pdf(pdf_path, output_dir=extracted_dir)
        texts       = load_extracted_text(extracted_dir)
        chunk_count = store_in_chromadb(texts, collection_name)

        # store session — each user gets their own
        sessions[session_id] = {
            "collection_name": collection_name,
            "pdf_name":        file.filename,
            "ready":           True,
            "history":         []   # per user conversation history
        }

        resp = JSONResponse({
            "status":     "ready",
            "filename":   file.filename,
            "chunks":     chunk_count,
            "collection": collection_name
        })
        resp.set_cookie("session_id", session_id, samesite="lax")
        return resp

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/ask")
async def ask(
    q: Question,
    session_id: Optional[str] = Cookie(None)
):
    # no session or no PDF uploaded yet
    if not session_id or session_id not in sessions:
        return {
            "answer":    "Please upload a PDF first.",
            "tool_used": "",
            "status":    "no_pdf"
        }

    user_session = sessions[session_id]

    if not user_session["ready"]:
        return {
            "answer":    "Please upload a PDF first.",
            "tool_used": "",
            "status":    "no_pdf"
        }

    try:
        answer, tool_used = run_agent(
            q.question,
            user_session["collection_name"],
            user_session["history"]   # pass per-user history
        )
        return {"answer": answer, "tool_used": tool_used, "status": "ok"}

    except Exception as e:
        return {"answer": f"Error: {str(e)}", "tool_used": "", "status": "error"}


@app.post("/clear")
async def clear(session_id: Optional[str] = Cookie(None)):
    if session_id and session_id in sessions:
        sessions[session_id]["history"] = []  # clear only this user's history
    return {"status": "cleared"}


@app.get("/status")
async def status(session_id: Optional[str] = Cookie(None)):
    if not session_id or session_id not in sessions:
        return {"ready": False}
    return sessions[session_id]


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)