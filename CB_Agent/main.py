from fastapi import FastAPI
from src.education_ai_system.api import embeddings_routes, crew_routes, docx_conversion_routes

app = FastAPI()

# Include routers
app.include_router(embeddings_routes.router, prefix="/api/embeddings", tags=["Embeddings"])
app.include_router(crew_routes.router, prefix="/api/crew", tags=["Crew"])
# app.include_router(docx_conversion_routes.router, prefix="/api/convert", tags=["DOCX Conversion"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
