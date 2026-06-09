from fastapi import FastAPI, UploadFile, File, Form
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

# per tab sessions
sessions = {}


class Question(BaseModel):
    question: str
    tab_id:   Optional[str] = None


class ClearRequest(BaseModel):
    tab_id: Optional[str] = None


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
        os.makedirs("/tmp/data",      exist_ok=True)
        os.makedirs("/tmp/extracted", exist_ok=True)

        pdf_path      = f"/tmp/data/{session_key}_{file.filename}"
        extracted_dir = f"/tmp/extracted/{session_key}"

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
async def ask(q: Question):
    session_key = q.tab_id

    if not session_key or session_key not in sessions:
        return {
            "answer":    "Please upload a PDF first.",
            "tool_used": "",
            "status":    "no_pdf"
        }

    user_session = sessions[session_key]

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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)