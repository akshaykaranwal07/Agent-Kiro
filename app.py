from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agent import run_agent, conversation_history
from extract import extract_text_from_pdf
from chunk_embed import load_extracted_text, store_in_chromadb
from dotenv import load_dotenv
import uvicorn
import shutil
import uuid
import os

load_dotenv()

app = FastAPI()

# session state
session = {
    "collection_name": None,
    "pdf_name": None,
    "ready": False
}


class Question(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # save uploaded PDF
        os.makedirs("data", exist_ok=True)
        os.makedirs("extracted", exist_ok=True)

        pdf_path = f"data/{file.filename}"
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # unique collection name per upload
        collection_name = f"pdf_{uuid.uuid4().hex[:8]}"

        print(f"Processing: {file.filename}")

        # extract text
        extract_text_from_pdf(pdf_path, output_dir="extracted")

        # chunk and embed
        texts = load_extracted_text("extracted")
        chunk_count = store_in_chromadb(texts, collection_name)

        # save to session
        session["collection_name"] = collection_name
        session["pdf_name"]        = file.filename
        session["ready"]           = True

        # clear conversation history for new PDF
        conversation_history.clear()

        return {
            "status":     "ready",
            "filename":   file.filename,
            "chunks":     chunk_count,
            "collection": collection_name
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/ask")
async def ask(q: Question):
    if not session["ready"]:
        return {
            "answer":   "Please upload a PDF first.",
            "tool_used": "",
            "status":   "no_pdf"
        }
    try:
        answer, tool_used = run_agent(q.question, session["collection_name"])
        return {"answer": answer, "tool_used": tool_used, "status": "ok"}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "tool_used": "", "status": "error"}


@app.post("/clear")
async def clear():
    conversation_history.clear()
    return {"status": "cleared"}


@app.get("/status")
async def status():
    return session


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)