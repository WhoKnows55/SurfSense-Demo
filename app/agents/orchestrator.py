"""
SurfSense Orchestrator

LLM-powered orchestrator that manages dialogue and delegates tasks to
three deterministic sub-agents through OpenAI function-calling.

This is the only component that calls the LLM directly.
"""

import json
from datetime import datetime, timezone

from app.agents.condition_agent import ConditionAssessmentAgent
from app.agents.forecast_data_agent import ForecastDataAgent
from app.agents.research_agent import ResearchAgent
from app.agents.trip_planning_agent import TripPlanningAgent
from app.core.llm_service import BaseLLMProvider
from app.core.logger import LoggerMixin, get_logger

logger = get_logger(__name__)


class Orchestrator(LoggerMixin):
    """LLM-powered orchestrator that delegates to deterministic sub-agents.

    Workflow:
    1. Maintains conversation history as an OpenAI message list.
    2. Collects all tool definitions from the three sub-agents.
    3. Calls Azure OpenAI with function-calling enabled.
    4. When the model returns a tool call, dispatches to the appropriate sub-agent.
    5. Feeds tool results back into the conversation and re-calls the model.
    6. Repeats until the model returns a text response.
    """

    _SYSTEM_PROMPT_TEMPLATE = """You are SurfSense, an AI surf trip planning assistant.

Current date and time (UTC): {current_datetime}

You help surfers plan trips by having a natural conversation to understand their needs,
then using your tools to fetch data, evaluate conditions, and build itineraries.

WORKFLOW:
1. Check what the user has already provided.
   - If spot AND skill level are both present in their message, skip to step 2 immediately
     — do NOT ask follow-up questions about dates, group size, or anything else.
   - If spot is missing, ask for it. If skill level is missing, ask for it. Ask for only
     what is missing, then proceed.
2. When the user mentions ANY surf spot or location, call research_spot first to gather
   information about it (coordinates, break type, hazards, skill levels, etc.).
3. Once you have spot data, call fetch_forecast for candidate spots.
4. Call assess_conditions to evaluate each spot for the user's skill level.
5. Call check_safety if any spots have concerning conditions.
6. Call find_surf_windows to identify the best times.
7. For multi-day trips, call plan_itinerary to optimise the schedule.
8. Present results in a clear, enthusiastic but safety-conscious manner.

RULES:
- Always call research_spot before fetch_forecast to get spot coordinates and metadata.
- If the user has already stated their skill level, never ask for it again.
- Never recommend spots rated "unsafe" for the user's level without explicit warnings.
- Be concise. Present key info first, details on request.
- If a tool call fails, explain the issue and suggest alternatives.
- When calling assess_conditions, ALWAYS explicitly pass the skill_level parameter
  matching what the user stated (beginner, intermediate, or advanced). Never omit it.
- When an assessment record contains a "feature_contributions" field, identify the
  top positive contributor and the top negative contributor by absolute magnitude and
  mention them in plain language — e.g. "Long swell period was the main positive factor;
  onshore wind reduced the score." Do not quote raw SHAP values.
- When any hours are rated unsafe, list every unsafe timestamp on its own line and
  include the word "unsafe" explicitly for each one — e.g. "13:00 — unsafe (waves 2.1 m,
  wind 45 kph)". Never aggregate multiple unsafe hours into a single range without
  naming each timestamp individually.
- Only recommend surf windows during daylight hours. Never suggest surfing at night.
- Never recommend surf windows or sessions that are in the past relative to the current
  date and time shown above.
"""

    @staticmethod
    def _build_system_prompt() -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return Orchestrator._SYSTEM_PROMPT_TEMPLATE.format(current_datetime=now)

    MAX_TOOL_ROUNDS = 10

    def __init__(self, llm_provider: BaseLLMProvider, settings):
        self._llm = llm_provider
        self._forecast_agent = ForecastDataAgent(settings)
        self._condition_agent = ConditionAssessmentAgent(settings)
        self._trip_agent = TripPlanningAgent()
        self._research_agent = ResearchAgent(llm_provider, settings)

        # Conversation history (OpenAI message format)
        self._messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

        # Tool dispatch table: tool_name -> (agent, method_name)
        self._tool_dispatch = {
            "research_spot": (self._research_agent, "research_spot"),
            "fetch_forecast": (self._forecast_agent, "fetch_forecast"),
            "fetch_contextual_info": (self._forecast_agent, "fetch_contextual_info"),
            "get_spot_metadata": (self._forecast_agent, "get_spot_metadata"),
            "assess_conditions": (self._condition_agent, "assess_conditions"),
            "check_safety": (self._condition_agent, "check_safety"),
            "get_skill_thresholds": (self._condition_agent, "get_skill_thresholds"),
            "find_surf_windows": (self._trip_agent, "find_surf_windows"),
            "plan_itinerary": (self._trip_agent, "plan_itinerary"),
            "rank_spots": (self._trip_agent, "rank_spots"),
        }

        # Collect all tool definitions
        self._tools = (
            ResearchAgent.get_tool_definitions()
            + ForecastDataAgent.get_tool_definitions()
            + ConditionAssessmentAgent.get_tool_definitions()
            + TripPlanningAgent.get_tool_definitions()
        )

        # Session data store for cross-tool dependencies
        self._session_data: dict[str, dict] = {}

    async def process(self, user_input: str) -> str:
        """Process user input through the orchestrator loop.

        1. Append user message to history.
        2. Call Azure OpenAI with tools.
        3. If model returns tool_calls, execute them and feed results back.
        4. Repeat until model returns a text response.
        5. Return the text response.
        """
        self._messages.append({"role": "user", "content": user_input})

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self._llm.chat_with_tools(self._messages, self._tools)
            message = response.choices[0].message

            # If no tool calls, we have a final text response
            if not message.tool_calls:
                content = message.content or ""
                self._messages.append({"role": "assistant", "content": content})
                return content

            # Process tool calls - append the assistant message with tool_calls
            self._messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                result = await self._execute_tool(tool_name, tool_args)

                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        return (
            "I've gathered the data but need to simplify my analysis. "
            "Let me summarise what I found..."
        )

    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Dispatch a tool call to the appropriate sub-agent."""
        if tool_name not in self._tool_dispatch:
            return {"error": f"Unknown tool: {tool_name}"}

        agent, method_name = self._tool_dispatch[tool_name]
        method = getattr(agent, method_name)

        try:
            enriched_args = self._enrich_args(tool_name, args)

            result = method(**enriched_args)
            # Handle async methods
            if hasattr(result, "__await__"):
                result = await result

            self._cache_result(tool_name, args, result)
            return result

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"error": str(e)}

    def _enrich_args(self, tool_name: str, args: dict) -> dict:
        """Inject cached session data into tool arguments where needed."""
        if tool_name == "assess_conditions":
            spot = args.get("spot_name", "")
            args = {k: v for k, v in args.items() if k != "spot_name"}
            if spot in self._session_data and "forecast" in self._session_data[spot]:
                args["forecast_data"] = self._session_data[spot]["forecast"]
        elif tool_name == "check_safety":
            spot = args.get("spot_name", "")
            if spot in self._session_data:
                if "forecast" in self._session_data[spot]:
                    args["forecast_data"] = self._session_data[spot]["forecast"]
                if "contextual" in self._session_data[spot]:
                    args["safety_info"] = self._session_data[spot][
                        "contextual"
                    ].get("safety", {})
        elif tool_name == "find_surf_windows":
            spot = args.get("spot_name", "")
            if (
                spot in self._session_data
                and "assessments" in self._session_data[spot]
            ):
                args["assessments"] = self._session_data[spot]["assessments"]
            if spot in self._session_data and "coordinates" in self._session_data[spot]:
                coords = self._session_data[spot]["coordinates"]
                if coords.get("lat") is not None:
                    args["latitude"] = coords["lat"]
                if coords.get("lon") is not None:
                    args["longitude"] = coords["lon"]
            args.pop("spot_name", None)  # method signature: (assessments, min_hours)
        elif tool_name == "plan_itinerary":
            spots_data = {}
            for spot_name in args.get("spot_names", []):
                if spot_name in self._session_data:
                    spots_data[spot_name] = self._session_data[spot_name]
            args["spots_data"] = spots_data
            args.pop("spot_names", None)
        elif tool_name == "rank_spots":
            spot_assessments = {}
            for spot_name in args.get("spot_names", []):
                if (
                    spot_name in self._session_data
                    and "assessments" in self._session_data[spot_name]
                ):
                    spot_assessments[spot_name] = self._session_data[
                        spot_name
                    ]["assessments"]
            args["spot_assessments"] = spot_assessments
        return args

    def _cache_result(self, tool_name: str, args: dict, result) -> None:
        """Store tool results in session cache for cross-tool dependencies."""
        if tool_name == "research_spot":
            # Cache research results keyed by spot name from the result
            spot = result.get("name") or args.get("query", "")
            if spot and isinstance(result, dict) and "error" not in result:
                if spot not in self._session_data:
                    self._session_data[spot] = {}
                self._session_data[spot]["research"] = result
                if result.get("latitude") and result.get("longitude"):
                    self._session_data[spot]["coordinates"] = {
                        "lat": result["latitude"],
                        "lon": result["longitude"],
                    }
                # Inject into ForecastDataAgent so it can resolve coordinates
                self._forecast_agent.set_research_data(spot, result)
                # Also register under the original query in case the LLM calls
                # fetch_forecast with the user's term rather than the official name.
                query = args.get("query", "")
                if query and query.lower().strip() != spot.lower().strip():
                    self._forecast_agent.set_research_data(query, result)
            return

        spot = args.get("spot_name", "")
        if not spot:
            return

        if spot not in self._session_data:
            self._session_data[spot] = {}

        if tool_name == "fetch_forecast":
            self._session_data[spot]["forecast"] = result
            if isinstance(result, dict) and "coordinates" in result:
                self._session_data[spot]["coordinates"] = result["coordinates"]
        elif tool_name == "fetch_contextual_info":
            self._session_data[spot]["contextual"] = result
        elif tool_name == "assess_conditions":
            self._session_data[spot]["assessments"] = result
        elif tool_name == "find_surf_windows":
            self._session_data[spot]["windows"] = result

    def reset(self) -> None:
        """Reset conversation and session data."""
        self._messages = [{"role": "system", "content": self._build_system_prompt()}]
        self._session_data = {}
