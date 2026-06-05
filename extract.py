# extract.py
# converts PDF pages to images and extracts text using Groq vision

from pdf2image import convert_from_path
from groq import Groq
from dotenv import load_dotenv
import base64
import os
import time
from config import PDF_PATH, EXTRACTED_DIR, POPPLER_PATH, DPI, VISION_MODEL

load_dotenv()
client = Groq()

def extract_text_from_pdf(pdf_path=PDF_PATH, output_dir=EXTRACTED_DIR):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting PDF to images — {pdf_path}")
    pages = convert_from_path(
        pdf_path,
        dpi=DPI,
        poppler_path=POPPLER_PATH
    )
    print(f"Total pages: {len(pages)}")

    all_text = []

    for i, page in enumerate(pages):
        print(f"Extracting page {i+1}/{len(pages)}...")

        img_path = f"{output_dir}/temp_page_{i+1}.jpg"
        page.save(img_path, "JPEG")

        try:
            with open(img_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
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
                    }
                ],
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

        time.sleep(4)  # groq free tier rate limit

    print(f"\nDone! {len(pages)} pages extracted to '{output_dir}'")
    return all_text


if __name__ == "__main__":
    extract_text_from_pdf()