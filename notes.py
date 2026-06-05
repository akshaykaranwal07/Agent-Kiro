from groq import Groq
from dotenv import load_dotenv
from pdf2image import convert_from_path
import base64
import os
import time

load_dotenv()

client = Groq()
POPPLER_PATH = r'C:\poppler\Library\bin'

def extract_text_from_pdf(pdf_path, output_dir="extracted"):
    os.makedirs(output_dir, exist_ok=True)

    print("Converting PDF pages to images...")
    pages = convert_from_path(
        pdf_path,
        dpi=250,
        poppler_path=POPPLER_PATH,
    )
    print(f"Total pages converted: {len(pages)}")

    all_text = []

    for i, page in enumerate(pages):
        print(f"Extracting page {i+1}/{len(pages)}...")

        # save as temp image
        img_path = f"extracted/temp_page_{i+1}.jpg"
        page.save(img_path, "JPEG")

        try:
            # convert to base64
            with open(img_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
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
                                "role": "system",
                                "content": """You are an English grammar tutor helping a student understand tenses.
                                Answer questions based only on the notes provided.
                                When answering:
                                - Always include the structure/formula if available in notes
                                - Always include example sentences if available
                                - If comparing tenses, list differences clearly
                                - Keep answers concise and student friendly
                                - If answer isn't in the notes, say so clearly"""
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )

            text = response.choices[0].message.content
            print(f"  Extracted {len(text)} characters")

        except Exception as e:
            print(f"  Failed on page {i+1}: {e}")
            text = f"[EXTRACTION FAILED FOR PAGE {i+1}]"

        all_text.append(text)

        with open(f"extracted/page_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        time.sleep(4)

    print(f"\nDone! {len(pages)} pages extracted.")
    return all_text


if __name__ == "__main__":
    text = extract_text_from_pdf("data/notes.pdf")