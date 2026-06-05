# app.py
# FastAPI backend — serves chat UI and handles questions

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import run_agent, conversation_history
import uvicorn

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r",encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
async def ask(q: Question):
    try:
        answer, tool_used = run_agent(q.question)
        return {"answer": answer,"tool_used": tool_used, "status": "ok"}
    except Exception as e:
        return {"answer": f"Error: {str(e)}","tool_used":"", "status": "error"}

@app.post("/clear")
async def clear():
    conversation_history.clear()
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)