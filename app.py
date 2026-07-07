from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
import base64
import io
import os

from PIL import Image

# Load .env
load_dotenv()

# Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")

# FastAPI app
app = FastAPI(title="Multimodal Image QA API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ImageRequest(BaseModel):
    image_base64: str
    question: str

# Home endpoint
@app.get("/")
def home():
    return {"message": "Image QA API is running"}

# Main endpoint
@app.post("/answer-image")
def answer_image(request: ImageRequest):

    # Decode Base64 Image
    image_bytes = base64.b64decode(request.image_base64)

    image = Image.open(io.BytesIO(image_bytes))

    prompt = f"""
You are an expert at reading receipts, invoices, tables, charts and scanned documents.

Question:
{request.question}

Rules:
1. Return ONLY the answer.
2. Never explain.
3. If numeric return only the number.
4. No currency symbols.
5. No units.
6. Output must be a single string.
"""

    response = model.generate_content(
        [prompt, image]
    )

    answer = response.text.strip()

    return {
        "answer": answer
    }