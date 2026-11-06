"""
SurfSense Agent Layer

Orchestrator-based architecture:
- Orchestrator: LLM-powered dialogue management via Azure OpenAI function-calling
- ResearchAgent: Dynamic surf spot research via Tavily web search
- ForecastDataAgent: Deterministic data aggregation sub-agent
- ConditionAssessmentAgent: Deterministic condition evaluation sub-agent
- TripPlanningAgent: Deterministic itinerary optimisation sub-agent
"""

from app.agents.orchestrator import Orchestrator
from app.agents.research_agent import ResearchAgent
from app.agents.forecast_data_agent import ForecastDataAgent
from app.agents.condition_agent import ConditionAssessmentAgent
from app.agents.trip_planning_agent import TripPlanningAgent

__all__ = [
    "Orchestrator",
    "ResearchAgent",
    "ForecastDataAgent",
    "ConditionAssessmentAgent",
    "TripPlanningAgent",
]
