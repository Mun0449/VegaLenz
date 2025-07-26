from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from PIL import Image
import io
import os
from typing import List  # 👈 다중 파일 처리를 위해 필요

# 🔐 환경변수로 서비스 계정 키 지정 (json 경로)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "veganlens-466807-b2d44b0fc9f7.json"

app = FastAPI()

# CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Vision 클라이언트
client = vision.ImageAnnotatorClient()

def extract_text(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    image_obj = vision.Image(content=img_byte_arr)
    response = client.document_text_detection(image=image_obj)

    return response.full_text_annotation.text

@app.post("/ocr")
async def run_ocr(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
            text = extract_text(image)
        except Exception as e:
            text = f"[ERROR: {str(e)}]"

        results.append({
            "filename": file.filename,
            "ocr_text": text
        })

    return JSONResponse({"results": results})