# src/education_ai_system/crew.py

import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import yaml
from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI
from typing import Dict, Any
from pydantic import SecretStr

from src.education_ai_system.tools.pinecone_exa_tools import PineconeRetrievalTool, ExaSearchContextualTool

# Load environment variables
load_dotenv()

# Define paths for configurations
agents_config_path = Path("src/education_ai_system/config/agents.yaml")
tasks_config_path = Path("src/education_ai_system/config/tasks.yaml")

# Helper function to load YAML config
def load_yaml(file_path: Path) -> Dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

# Load configurations
agents_config = load_yaml(agents_config_path)
tasks_config = load_yaml(tasks_config_path)

# Initialize the LLM with API Key
# def llm() -> ChatOpenAI:
#     api_key = os.getenv("OPENAI_API_KEY")
#     if api_key is None:
#         raise ValueError("OPENAI_API_KEY is not set.")
#     return ChatOpenAI(api_key=SecretStr(api_key), model='GPT-4o')

# Initialize tools
pinecone_tool = PineconeRetrievalTool()
exa_search_tool = ExaSearchContextualTool()

# Define agent creation functions
def create_agent(agent_key: str, tools) -> Agent:
    config = agents_config.get(agent_key)
    if config is None:
        raise ValueError(f"Agent configuration for '{agent_key}' not found in agents.yaml.")
    
    return Agent(
        role=config.get("role", "Unknown Role"),
        goal=config.get("goal", "Undefined Goal"),
        backstory=config.get("backstory", "No backstory available."),
        tools=tools,
        verbose=True,
        #llm=llm()
    )

# Define task creation functions
def create_task(task_key: str, agent: Agent, inputs: Dict[str, Any]) -> Task:
    config = tasks_config.get(task_key, {})
    # Convert dictionary entries into Task instances if feasible
    context = [Task(description=key, expected_output=value, agent=agent) for key, value in inputs.items()]
    return Task(
        description=config.get("description", "No description."),
        expected_output=config.get("expected_output", "No expected output."),
        agent=agent,
        context=context,
        output_file=generate_log_filename(task_key)
    )


# Log filename generation
def generate_log_filename(task_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f'logs/{timestamp}_{task_name}_task.md'

# Define Crew with dependencies
def build_education_crew() -> Crew:
    shared_context = {}

    # Create agents
    context_retrieval_agent = create_agent("context_retrieval_agent", [pinecone_tool])
    scheme_of_work_agent = create_agent("scheme_of_work_generator", [exa_search_tool])
    lesson_plan_agent = create_agent("lesson_plan_generator", [exa_search_tool])
    lesson_notes_agent = create_agent("lesson_notes_generator", [exa_search_tool])

    # Define tasks with dependency on shared context
    context_retrieval_task = create_task("context_retrieval_task", context_retrieval_agent, shared_context)
    scheme_of_work_task = create_task("scheme_of_work_task", scheme_of_work_agent, shared_context)
    lesson_plan_task = create_task("lesson_plan_task", lesson_plan_agent, shared_context)
    lesson_notes_task = create_task("lesson_notes_task", lesson_notes_agent, shared_context)

    # Sequential crew setup
    crew = Crew(
        agents=[context_retrieval_agent, scheme_of_work_agent, lesson_plan_agent, lesson_notes_agent],
        tasks=[context_retrieval_task, scheme_of_work_task, lesson_plan_task, lesson_notes_task],
        process=Process.sequential,
        verbose=True
    )

    return crew

