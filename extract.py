import fitz
import base64
import os
import time
from google import genai
from dotenv import load_dotenv
from config import VISION_MODEL

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(VISION_MODEL)

def extract_text_from_pdf(pdf_path, output_dir="extracted"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting PDF to images — {pdf_path}")
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")

    all_text = []

    for i, page in enumerate(doc):
        print(f"Extracting page {i+1}/{len(doc)}...")

        # try direct text extraction first — saves tokens
        text = page.get_text().strip()

        if len(text) > 50:
            print(f"  Page {i+1} — direct text extraction")
        else:
            # scanned or handwritten — use vision
            print(f"  Page {i+1} — using Gemini vision")
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_path = f"{output_dir}/temp_page_{i+1}.jpg"
            pix.save(img_path)

            try:
                with open(img_path, "rb") as f:
                    image_data = f.read()

                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": image_data},
                    """You are an expert at reading documents.
                    Transcribe ALL text visible in this image.
                    Rules:
                    - Preserve structure — headings, subheadings, bullet points, tables
                    - If text is unclear, make your best guess and mark it with [?]
                    - Preserve formulas, diagram labels, and special symbols
                    - Only return the transcribed text, nothing else
                    - If page is blank write BLANK PAGE"""
                ])

                text = response.text
                print(f"  {len(text)} characters extracted")
                time.sleep(4)  # gemini free tier rate limit

            except Exception as e:
                print(f"  Failed on page {i+1}: {e}")
                text = f"[EXTRACTION FAILED FOR PAGE {i+1}]"

        all_text.append(text)

        with open(f"{output_dir}/page_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    page_count = len(doc)
    doc.close()
    print(f"\nDone! {page_count} pages extracted.")
    return all_text


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/notes.pdf"
    extract_text_from_pdf(pdf)