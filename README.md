# Vanilla RAG Pipeline

Chat with your handwritten notes using AI.
Built from scratch — no LangChain, no shortcuts.

## Stack
- Groq Vision — handwriting extraction
- Sentence Transformers — embeddings  
- ChromaDB — vector storage
- Groq LLaMA 3.3 — answer generation

## How it works
1. PDF pages converted to images via Poppler
2. Groq vision reads handwritten text page by page
3. Text chunked by concept blocks
4. Chunks embedded and stored in ChromaDB
5. Questions answered using semantic retrieval + LLM

## Project Structure
├── data/               # put your PDF here (not tracked)
├── extracted/          # OCR output pages (not tracked)
├── chroma_db/          # vector database (not tracked)
├── extract.py          # PDF → text via Groq vision
├── chunk_embed.py      # chunk → embed → store in ChromaDB
├── query.py            # ask questions against your notes
├── eval.py             # evaluate RAG accuracy
└── requirements.txt    # dependencies

## Setup
1. Clone the repo
2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate
3. Install dependencies
   python -m pip install -r requirements.txt
4. Add API keys to .env
   GROQ_API_KEY=your-key
   GEMINI_API_KEY=your-key
5. Add your PDF to data/ folder

## Run
python extract.py        # extract text from PDF
python chunk_embed.py    # chunk and embed
python query.py          # start chatting
python eval.py           # run evaluation