import fitz  # pymupdf — no poppler needed
import base64
import os
import time
from groq import Groq
from dotenv import load_dotenv
from config import VISION_MODEL

load_dotenv()
client = Groq()

def extract_text_from_pdf(pdf_path, output_dir="extracted"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting PDF to images — {pdf_path}")
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")

    all_text = []

    for i, page in enumerate(doc):
        print(f"Extracting page {i+1}/{len(doc)}...")

        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_path = f"{output_dir}/temp_page_{i+1}.jpg"
        pix.save(img_path)

        try:
            with open(img_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": """You are an expert at reading documents.
                            Transcribe ALL text visible in this image.
                            Rules:
                            - Preserve structure — headings, subheadings, bullet points, tables
                            - If text is unclear, make your best guess and mark it with [?]
                            - Preserve formulas, diagram labels, and special symbols
                            - Only return the transcribed text, nothing else
                            - If page is blank write BLANK PAGE"""
                        }
                    ]
                }],
                max_tokens=2000
            )
            text = response.choices[0].message.content
            print(f"  {len(text)} characters extracted")

        except Exception as e:
            print(f"  Failed on page {i+1}: {e}")
            text = f"[EXTRACTION FAILED FOR PAGE {i+1}]"

        all_text.append(text)

        with open(f"{output_dir}/page_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        time.sleep(4)

    doc.close()
    print(f"\nDone! {len(doc)} pages extracted.")
    return all_text


if __name__ == "__main__":
    extract_text_from_pdf("data/notes.pdf")