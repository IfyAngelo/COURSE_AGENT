# src/education_ai_system/embeddings/pinecone_manager.py

import os
from pinecone import Pinecone
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = 'seismic-agent'

def upsert_to_pinecone(chunks, metadata):
    index = pc.Index(index_name)
    embeddings_with_metadata = []

    for chunk, meta in zip(chunks, metadata):
        # Generate embedding
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

        # Combine embedding and metadata
        metadata_dict = {
            "subject": meta[0],
            "grade_level": meta[1],
            "text_chunk": chunk
        }

        # Append embedding and metadata
        embeddings_with_metadata.append((embedding[0], metadata_dict))

    # Prepare upsert data with IDs, embeddings, and metadata
    ids = [f"chunk-{i}" for i in range(len(embeddings_with_metadata))]
    vectors_to_upsert = [(ids[i], embeddings_with_metadata[i][0], embeddings_with_metadata[i][1]) for i in range(len(embeddings_with_metadata))]

    # Upsert to Pinecone
    index.upsert(vectors_to_upsert)