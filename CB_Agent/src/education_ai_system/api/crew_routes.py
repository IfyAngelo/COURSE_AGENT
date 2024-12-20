from fastapi import APIRouter, Body
from src.education_ai_system.main import run_and_save_crew_output

router = APIRouter()

@router.post("/run_agents")
async def run_agents(inputs: dict = Body(...)):
    # Run crew agents and capture the output
    result = run_and_save_crew_output(inputs)

    return result
