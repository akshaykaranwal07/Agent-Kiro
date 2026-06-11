import fitz
import base64
import os
import time
from groq import Groq
from dotenv import load_dotenv
from config import VISION_MODEL

load_dotenv()

client = Groq()


def extract_page_with_vision(image_path: str, max_retries: int = 3) -> str:
    """Use Groq vision for scanned/handwritten pages with an automatic retry loop"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Retry loop to handle unexpected API drops or rate limits
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
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
                            "text": """Transcribe ALL text visible in this image.
                            - Preserve structure — headings, bullets, tables
                            - If text is unclear mark it with [?]
                            - Only return transcribed text, nothing else
                            - If blank write BLANK PAGE"""
                        }
                    ]
                }],
                max_tokens=2000
            )
            # If successful, return the text immediately and break the retry loop
            return response.choices[0].message.content

        except Exception as e:
            print(f"      [Attempt {attempt + 1}/{max_retries}] Vision API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)  # Wait 3 seconds before trying again
            else:
                # If all retries fail, raise the exception up to the main function
                raise e

def extract_text_from_pdf(pdf_path: str, output_dir: str = "extracted") -> list:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening: {pdf_path}")
    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"Total pages: {total}")

    all_text = []

    for i, page in enumerate(doc):
        print(f"Page {i+1}/{total}...")

        text = page.get_text().strip()
        print(f"  raw text length: {len(text)}")

        if len(text) > 10:
            print(f"  direct extraction — {len(text)} chars")

        else:
            print(f"  vision OCR...")
            mat      = fitz.Matrix(2.0, 2.0)
            pix      = page.get_pixmap(matrix=mat)
            img_path = f"{output_dir}/temp_page_{i+1}.jpg"
            pix.save(img_path)

            try:
                text = extract_page_with_vision(img_path)
                print(f"  vision done — {len(text)} chars")
                time.sleep(2)

            except Exception as e:
                print(f"  vision failed: {e}")
                text = f"[EXTRACTION FAILED FOR PAGE {i+1}]"

            # cleanup INSIDE else — only runs when vision was used
            if os.path.exists(img_path):
                os.remove(img_path)

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