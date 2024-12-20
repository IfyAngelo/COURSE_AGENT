# from fastapi import APIRouter, HTTPException
# from fastapi.responses import FileResponse
# from src.education_ai_system.utils.file_operations import convert_md_to_docx, load_inputs
# from pathlib import Path

# router = APIRouter()

# @router.post("/convert_md_to_docx")
# async def convert_and_download_docx():
#     # Load inputs to name the file
#     inputs = load_inputs()
#     subject = inputs['subject'].replace(" ", "_").lower()
#     grade_level = inputs['grade_level'].replace(" ", "_").lower()
#     topic = inputs['topic'].replace(" ", "_").lower()

#     # Convert markdown files in logs to DOCX
#     convert_md_to_docx(subject, grade_level, topic)

#     # Locate DOCX output file
#     output_file = Path(f"genai_output/{subject}_{grade_level}_{topic}_context_retrieval_agent.docx")
#     if not output_file.exists():
#         raise HTTPException(status_code=404, detail="File not found")

#     return FileResponse(path=output_file, filename=output_file.name, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
