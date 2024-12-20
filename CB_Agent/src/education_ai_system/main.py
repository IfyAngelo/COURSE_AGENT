from src.education_ai_system.crew import build_education_crew
from src.education_ai_system.embeddings.pinecone_manager import upsert_to_pinecone
from src.education_ai_system.data_processing.pdf_extractor import extract_text_and_tables
from src.education_ai_system.data_processing.text_chunker import split_text_into_chunks
from src.education_ai_system.data_processing.metadata_extractor import extract_metadata
from src.education_ai_system.utils.validators import validate_user_input, load_predefined_inputs
import os
import json


def process_and_index_pdf(pdf_path):
    """
    Processes a PDF file and indexes its content in Pinecone.
    Args:
        pdf_path (str): Path to the PDF file.
    """
    print("Processing and indexing PDF...")
    text, tables = extract_text_and_tables(pdf_path)  # Extract text and tables
    chunks = split_text_into_chunks(text)  # Chunk extracted text
    metadata = [extract_metadata(chunk) for chunk in chunks]  # Extract metadata
    upsert_to_pinecone(chunks, metadata)  # Index in Pinecone
    print("PDF processing and indexing complete.")


def get_user_input() -> str:
    """
    Collects user input as a plain string for subject, grade level, and topic.
    Returns:
        str: User query in the format 'subject, grade_level, topic'.
    """
    return input("Enter your query (e.g., Civic Education, Primary Two, National Symbols): ").strip()


def run_and_save_crew_output(query: str, predefined_inputs: dict):
    """
    Runs the education crew and saves the output.
    Args:
        query (str): User query as a plain string in the format 'subject, grade_level, topic'.
        predefined_inputs (dict): Predefined valid inputs for validation.
    Returns:
        dict: Crew execution results.
    """
    # Parse the string query into a dictionary
    parts = query.split(",")
    if len(parts) != 3:
        print("Invalid query format. Must be 'subject, grade_level, topic'.")
        return {"status": "invalid", "message": "Invalid query format."}

    query_dict = {
        "subject": parts[0].strip().lower(),
        "grade_level": parts[1].strip().lower(),
        "topic": parts[2].strip().lower(),
    }

    # Validate user input
    if not validate_user_input(query_dict, predefined_inputs):
        print("Invalid input. Use predefined subjects, grade levels, and topics.")
        return {"status": "invalid", "message": "Invalid input. Use predefined values."}

    print("Input validated. Proceeding with crew execution...")

    # Build and execute the Crew
    crew = build_education_crew()
    result = crew.kickoff(inputs=query_dict)

    return result


def main():
    predefined_inputs = load_predefined_inputs()

    pdf_path = '/Users/libertyelectronics/Desktop/curriculum_builder/CB_Agent/data/pri1-3_civic (1).pdf'
    process_and_index_pdf(pdf_path)

    query = get_user_input()  # Get user input
    with open("user_inputs.json", "w") as f:
        json.dump(query, f)  # Save for debugging

    result = run_and_save_crew_output(query, predefined_inputs)

    if result is None:
        print("Crew execution failed due to invalid inputs.")
    else:
        # Check if the `CrewOutput` object contains a `status` attribute
        if hasattr(result, "status") and getattr(result, "status") == "invalid":
            print("Crew execution failed due to invalid inputs.")
        else:
            print("Crew execution completed successfully.")
            
            # Print all attributes of `CrewOutput` for debugging
            print("Crew Output:")
            print(result)


if __name__ == "__main__":
    main()





