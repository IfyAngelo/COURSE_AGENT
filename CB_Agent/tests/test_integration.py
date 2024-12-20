# tests/test_integration.py
from src.education_ai_system.main import main

def test_main_execution():
    assert main() == "output.docx"
