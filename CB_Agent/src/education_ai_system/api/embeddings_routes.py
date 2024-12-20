from fastapi import APIRouter, UploadFile, File
from src.education_ai_system.main import process_and_index_pdf
import os
router = APIRouter()

@router.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    # Save uploaded file locally
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Process and store embeddings
    process_and_index_pdf(file_path)

    # Clean up temporary file
    os.remove(file_path)
    return {"message": "PDF processed and embeddings stored successfully."}
