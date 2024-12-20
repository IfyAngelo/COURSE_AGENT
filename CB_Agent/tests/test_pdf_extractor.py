# tests/test_pdf_extractor.py
from src.project_name.data_processing.pdf_extractor import extract_text_and_tables

def test_extract_text_and_tables():
    text, tables = extract_text_and_tables('sample.pdf')
    assert isinstance(text, str)
    assert isinstance(tables, list)
