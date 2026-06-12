# Kiro — Your Personal AI Knowledge Base

> Upload anything. Ask anything. Actually learn from it.

Kiro is an agentic RAG application built from scratch — no LangChain, no shortcuts. Chat with any PDF, get answers from the web, and watch it get smarter with every improvement.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green?style=flat)
![Deployed](https://img.shields.io/badge/Deployed-Railway-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## Live Demo

🔗 **[https://agent-kiro-production.up.railway.app/]**

---

## What Kiro Does
Upload any PDF → Kiro reads it, chunks it, embeds it

Ask a question → Kiro decides: search your document or the web?

Get an answer  → grounded, accurate, cited

Works on typed PDFs, scanned documents, and handwritten notes.

---

## How It Works
Upload PDF

↓

PyMuPDF extracts text directly (typed PDFs)

Groq LLaMA-4 Vision reads scanned/handwritten pages

↓

Structure-aware chunking by concept blocks

↓

Sentence Transformers embed chunks → ChromaDB stores vectors

↓

User asks question

↓

Query rewriting → vague questions become specific search queries

↓

Agent routes: search notes OR search web

↓

Retrieved chunks reranked by relevance

↓

Groq LLaMA generates grounded answer

↓

Answer with source attribution

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

## Engineering Layers

### ✅ Layer 1 — Core RAG Pipeline
PDF ingestion → smart extraction (direct text + vision OCR fallback)

Structure-aware chunking → concept block splitting

Dense embeddings → ChromaDB vector storage

Semantic retrieval → cosine similarity search

Agentic routing → LLM decides notes vs web

Conversation memory → per-user session history

### ✅ Layer 2 — Prompt Engineering
Versioned prompt system → every prompt change tracked (v1.0+)

Query rewriting → transforms vague questions before retrieval

Personal signal detection → protects document queries from over-rewriting

Chain of thought → auto-detected for complex comparison questions

Temperature control → 0.0 routing (deterministic), 0.1 answers (precise)

Structured output prompts → consistent answer formatting

### ✅ Layer 3 — Evaluation Framework
General eval suite → tests behavior not content, works on any PDF

4 test categories → factual, grounding, web routing, instruction following

3 eval types → exact match, contains, LLM-as-judge

Category-based scoring → pinpoints exactly what to improve

Baseline tracking → 80% accuracy on v1.0 prompts

### ✅ Layer 4 — Sessions and Multi-User
Tab-level sessions → sessionStorage per browser tab

Per-user isolation → each tab has own PDF and conversation history

No shared state → laptop and mobile fully independent

Graceful degradation → web search works without PDF upload

### 🔄 Layer 5 — Context Engineering 
Metadata filtering → chunks know their page number

Reranking → retrieved chunks reordered by relevance

Lost in the middle fix → best chunks at start and end of context

Parent-child chunking → retrieve small, send parent for context

Hybrid search → semantic + keyword (BM25) combined

### ⬜ Layer 6 — Caching and Reliability
Semantic caching → similar questions return cached answers

Exponential backoff → rate limit handling with retries

Model fallback chain → Groq fails → Gemini fallback

Request timeout handling → user-friendly error on slow responses

Token usage tracking → cost per query logged

### ⬜ Layer 7 — Observability
Langfuse tracing → every LLM call, tool call, retrieval step visible

Structured logging → question, tool used, answer, latency per request

Latency tracking → time each pipeline step

Error tracking → all failures caught and logged with context

### ⬜ Layer 8 — Advanced RAG
Query rewriting V2 → multi-query retrieval (3 versions, merge results)

HyDE → hypothetical document embeddings for better chunk matching

Self-RAG → model decides when to retrieve vs answer from memory

RAGAS evaluation → faithfulness, relevance, recall metrics

Automated regression testing → eval runs on every deploy

### ⬜ Layer 9 — Product Features
Streaming responses → answers appear word by word

Multi-document support → upload multiple PDFs, search across all

YouTube + URL ingestion → chat with any content not just PDFs

Quiz mode → Kiro asks YOU questions from your material

Flashcard generation → auto-create study cards from document

Progress tracking → what you know, what you don't

### ⬜ Layer 10 — Production Hardening
User authentication → accounts with persistent sessions

CI/CD → GitHub Actions auto-test and deploy on push

Docker optimization → smaller image, faster cold start

Rate limiting → per-user request limits

Cost optimization → token usage minimization

---

## Local Setup

### Prerequisites
- Python 3.11+
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Installation

```bash
git clone https://github.com/reloaddd/Agent-Kiro.git
cd Vanillia-RAG

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Environment Variables

```bash
# .env
GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key
EVAL_COLLECTION=tenses_notes
```

### Run

```bash
python app.py
```

Open `http://localhost:8000`

---

## Project Structure
vanilla-RAG/

│

├── templates/

│   └── index.html          # Kiro chat UI

│

├── app.py                  # FastAPI — upload, ask, sessions

├── agent.py                # routing agent + memory

├── tools.py                # search_notes + search_web

├── extract.py              # PDF → text (direct + vision)

├── chunk_embed.py          # chunking + embeddings + ChromaDB

├── prompts.py              # versioned prompts (v1.0)

├── eval.py                 # general eval suite

├── config.py               # configuration

│

├── Dockerfile              # Railway deployment

├── railway.toml            # Railway config

├── requirements.txt

└── .gitignore

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Kiro chat UI |
| POST | `/upload` | Upload and process PDF |
| POST | `/ask` | Ask a question |
| POST | `/clear` | Clear conversation history |
| GET | `/status` | Active session count |

---

## Eval Results

| Version | Score | Notes |
|---|---|---|
| baseline | 70% | vanilla retrieval |
| v1.0 | 80% | prompt engineering + query rewriting |
| v1.1 | TBD | context engineering + reranking |
| v2.0 | TBD | advanced RAG + HyDE |

---

## Deployment

### Railway (one click)

1. Fork this repo
2. Connect to [railway.app](https://railway.app)
3. Add environment variables
4. Deploy — Railway detects Dockerfile automatically

### Docker

```bash
docker build -t kiro .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your-key \
  kiro
```

---

## What I Learned Building This
→ RAG pipeline from scratch — no frameworks, understand every line

→ Vision OCR for handwritten and scanned documents

→ Agentic tool routing — LLM decides what to do next

→ Prompt engineering that's measurable not just felt

→ Why evals matter — adding query rewriting dropped score 30%

→ Per-user session isolation — production multi-user patterns

→ Full stack AI deployment — FastAPI, Docker, Railway

→ The difference between "it feels better" and "it scores better"

---

## Roadmap

- [x] Vanilla RAG pipeline
- [x] Vision OCR
- [x] Agentic routing
- [x] Prompt engineering + query rewriting
- [x] Eval framework
- [x] Per-user sessions
- [ ] Context engineering + reranking
- [ ] Semantic caching
- [ ] Observability with Langfuse
- [ ] Advanced RAG (HyDE, Self-RAG)
- [ ] Streaming responses
- [ ] Multi-document support
- [ ] Quiz mode + flashcards
- [ ] User authentication

---

## Contributing

Pull requests welcome. Open an issue first for major changes.

---

## License

MIT — free to use, modify, distribute.

---

<p align="center">
Built by <a href="https://github.com/reloaddd">Akshay</a> · AI Engineer in the making · Building in public
</p>