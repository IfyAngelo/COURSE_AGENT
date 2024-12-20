# import yaml
# from typing import Dict

# def load_predefined_inputs() -> Dict:
#     with open("src/education_ai_system/config/predefined_inputs.yaml", "r") as f:
#         return yaml.safe_load(f)

# predefined_inputs = load_predefined_inputs()

# def validate_user_input(query: Dict[str, str], predefined_inputs: Dict) -> bool:
#     # Normalize input for case-insensitivity and spacing
#     query = {key: value.strip().lower() for key, value in query.items()}

#     # Validate subject
#     subject = next((s for s in predefined_inputs["subjects"] if s["name"].strip().lower() == query["subject"]), None)
#     if not subject:
#         return False

#     # Validate grade level
#     grade_level = next((g for g in subject["grade_levels"] if g["name"].strip().lower() == query["grade_level"]), None)
#     if not grade_level:
#         return False

#     # Validate topic
#     return query["topic"] in [t.strip().lower() for t in grade_level["topics"]]

import yaml
from typing import Dict, Optional

def load_predefined_inputs() -> Dict:
    with open("src/education_ai_system/config/predefined_inputs.yaml", "r") as f:
        return yaml.safe_load(f)

predefined_inputs = load_predefined_inputs()

def parse_query(query: str) -> Optional[Dict[str, str]]:
    """
    Parses a plain string query into a structured dictionary.
    Args:
        query (str): Plain string query in the format 'subject, grade_level, topic'.
    Returns:
        dict: Parsed query with 'subject', 'grade_level', and 'topic' keys, or None if parsing fails.
    """
    parts = query.split(",")
    if len(parts) != 3:
        return None

    return {
        "subject": parts[0].strip().lower(),
        "grade_level": parts[1].strip().lower(),
        "topic": parts[2].strip().lower()
    }

def validate_user_input(query: Dict[str, str], predefined_inputs: Dict) -> bool:
    """
    Validates user input query against predefined subjects, grade levels, and topics.
    Args:
        query (dict): User input with keys 'subject', 'grade_level', 'topic'.
        predefined_inputs (dict): Predefined valid inputs for validation.
    Returns:
        bool: True if the query is valid, False otherwise.
    """
    # Normalize input for case-insensitivity and spacing
    query = {key: value.strip().lower() for key, value in query.items()}

    # Validate subject
    subject = next((s for s in predefined_inputs["subjects"] if s["name"].strip().lower() == query["subject"]), None)
    if not subject:
        return False

    # Validate grade level
    grade_level = next((g for g in subject["grade_levels"] if g["name"].strip().lower() == query["grade_level"]), None)
    if not grade_level:
        return False

    # Validate topic
    return query["topic"] in [t.strip().lower() for t in grade_level["topics"]]
