import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'veganlens-466807-b2d44b0fc9f7.json'
# app/veganLens.py
from google.cloud import vision
import io
from PIL import Image

client = vision.ImageAnnotatorClient()

def extract_text(image: Image.Image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    image_obj = vision.Image(content=img_byte_arr)

    response = client.document_text_detection(image=image_obj)
    return response.full_text_annotation.text
# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

from app.veganLens import extract_text

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keys/vision_key.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    text = extract_text(image)
    return JSONResponse({"ocr_text": text})
