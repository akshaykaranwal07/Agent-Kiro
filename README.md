# Kiro — Chat With Any Document

Kiro is an agentic RAG (Retrieval-Augmented Generation) application that lets you upload any PDF and have an intelligent conversation with it. It combines semantic search over your document with live web search to answer any question.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Deployed](https://img.shields.io/badge/Deployed-Railway-purple)

---

## Live Demo

🔗 [your-railway-url.up.railway.app](https://agent-kiro-production.up.railway.app/)

---

## Features

- **Upload any PDF** — handwritten notes, textbooks, research papers, resumes
- **Agentic routing** — automatically decides whether to search your document or the web
- **Vision-based OCR** — reads handwritten and scanned PDFs using Groq LLaMA-4 Vision
- **Semantic search** — finds relevant content by meaning, not just keywords
- **Conversation memory** — remembers context across follow-up questions
- **Clean chat UI** — sidebar layout, suggested questions, source attribution
- **Mobile responsive** — works on any device

---

## Demo

Upload any PDF → Ask anything → Kiro decides where to look

"What are the main topics in this document?"  →  searches your PDF
"Who is the current Prime Minister of India?" →  searches the web
"Give me examples of what you just explained" →  remembers context

---

## How It Works

Upload PDF
↓
PyMuPDF converts pages to images
↓
Groq LLaMA-4 Vision extracts text (handles handwriting)
↓
Text split into concept-aware chunks
↓
Chunks embedded using Sentence Transformers
↓
Stored in ChromaDB vector database
↓
Agent routes your question:
├── ChromaDB  → if question is about your document
└── Web search → if question needs current information
↓
Groq LLaMA 3.3 generates grounded answer

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11 |
| LLM | Groq LLaMA 3.3 70B |
| Vision / OCR | Groq LLaMA 4 Scout 17B |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| Web Search | DuckDuckGo Search |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Deployment | Railway (Docker) |

---

## Local Setup

### Prerequisites

- Python 3.11+
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Installation

```bash
# clone the repo
git clone https://github.com/yourusername/vanilla-RAG.git
cd vanilla-RAG

# create virtual environment
python -m venv venv

# activate — Windows
venv\Scripts\activate

# activate — Mac/Linux
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:
GROQ_API_KEY=your-groq-api-key-here

### Run

```bash
python app.py
```

Open `http://localhost:8000`

---

## Usage

1. Open Kiro in your browser
2. Upload any PDF using the sidebar — click or drag and drop
3. Wait for processing (speed depends on PDF size and pages)
4. Ask anything about your document or any general question
5. Kiro automatically routes to the right source and answers

---

## Project Structure
vanilla-RAG/
│
├── templates/
│   └── index.html          # chat UI — sidebar layout, mobile responsive
│
├── app.py                  # FastAPI server — upload, ask, clear endpoints
├── agent.py                # routing agent — tool selection + memory
├── tools.py                # search_notes and search_web tools
├── extract.py              # PDF → images → text via Groq vision
├── chunk_embed.py          # chunking + embeddings + ChromaDB storage
├── config.py               # central configuration
│
├── Dockerfile              # production Docker config
├── railway.toml            # Railway deployment config
├── requirements.txt        # Python dependencies
└── .gitignore

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves chat UI |
| POST | `/upload` | Upload and process a PDF file |
| POST | `/ask` | Send a question, get an answer |
| POST | `/clear` | Clear conversation history |
| GET | `/status` | Check current session status |

### Example

```bash
# upload a PDF
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"

# ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics?"}'
```

---

## Deployment

Kiro is production-ready and deploys on Railway with zero configuration.

### Deploy on Railway

1. Fork this repository
2. Go to [railway.app](https://railway.app) and create a new project
3. Connect your forked GitHub repository
4. Add environment variable: `GROQ_API_KEY`
5. Railway auto-detects the Dockerfile and deploys

### Deploy with Docker

```bash
docker build -t kiro .
docker run -p 8000:8000 -e GROQ_API_KEY=your-key kiro
```

---

## What I Learned Building This

- Built a complete RAG pipeline from scratch — no LangChain, no shortcuts
- Understood why things fail — chunking, retrieval quality, context limits
- Implemented agentic tool routing — how LLMs decide which tool to use
- Vision-based OCR pipeline for handwritten document processing
- Vector similarity search, embedding models, and semantic retrieval
- RAG evaluation using LLM-as-judge scoring (achieved 90% accuracy)
- Full-stack AI application — FastAPI backend, vanilla JS frontend
- Production deployment with Docker on Railway

---

## Roadmap

- [ ] Streaming responses (word by word like ChatGPT)
- [ ] Multiple concurrent PDF sessions
- [ ] User authentication
- [ ] Parent-child chunk retrieval for better accuracy
- [ ] Support PDF (work in progress for other formats)
- [ ] Conversation history persistence

---

## Contributing

Pull requests are welcome. For major changes please open an issue first.

---

## License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built by <a href="https://github.com/reloaddd">Reload</a> — AI Engineer</p>
