# config.py
# ── only file you need to edit ──────────────────────────
PDF_NAME = "notes"  # change this to your PDF filename (without .pdf)
# ────────────────────────────────────────────────────────

import os

#PDF_PATH        = f"data/{PDF_NAME}.pdf"
#COLLECTION_NAME = PDF_NAME.lower().replace(" ", "_")
EXTRACTED_DIR   = "extracted"
CHROMA_PATH     = "./chroma_db"
POPPLER_PATH    = r"C:\poppler\Library\bin"
DPI             = 250
CHUNK_MIN_WORDS = 10
CHUNK_MAX_WORDS = 200
N_RESULTS       = 5
LLM_MODEL       = "llama-3.3-70b-versatile"
VISION_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"