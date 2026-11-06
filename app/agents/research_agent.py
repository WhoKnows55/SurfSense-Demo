"""
SurfSense Research Agent

Dynamically researches surf spots via Tavily web search + LLM extraction.
Replaces the static spot database — any spot worldwide can be looked up
at conversation time.
"""

import json
import re
from typing import Any

_LAT_RE = re.compile(r"[Ll]at(?:itude)?[.\s:]+(-?\d{1,3}\.\d+)", re.IGNORECASE)
_LON_RE = re.compile(r"[Ll]on(?:gitude)?[.\s:]+(-?\d{1,3}\.\d+)", re.IGNORECASE)

# Canonical coordinates for known spots — applied as a hard override after LLM
# extraction so that stochastic Tavily results never cause wrong-location fetches.
# Keyed by lowercase query fragment; first match wins.
_KNOWN_COORDS: dict[str, tuple[float, float]] = {
    "sagres": (37.0097, -8.9372),
    "tonel": (37.0097, -8.9372),
    "guincho": (38.7009, -9.4745),
    "ericeira": (38.9617, -9.4164),
    "peniche": (39.3561, -9.3812),
    "supertubos": (39.3561, -9.3812),
}

from app.core.logger import LoggerMixin, get_logger

logger = get_logger(__name__)

# Schema the LLM must produce when extracting spot info from search results.
EXTRACTION_PROMPT = """\
You are a surf-spot data extraction assistant.

Given the following web search results about a surf spot, extract structured
information and return ONLY a valid JSON object (no markdown, no explanation).

If a field cannot be determined from the search results, use null.

Required JSON schema:
{{
  "name": "<official spot name>",
  "latitude": <float>,
  "longitude": <float>,
  "region": "<geographic region>",
  "country": "<country>",
  "timezone": "<IANA timezone, e.g. Europe/Lisbon>",
  "break_type": "<beach | reef | point | rivermouth>",
  "wave_direction": "<left | right | right_and_left>",
  "bottom": "<sand | rock | reef | cobblestone>",
  "crowd_level": "<low | medium | high | very_high>",
  "skill_minimum": "<beginner | intermediate | advanced | expert>",
  "skill_recommended": "<beginner | intermediate | advanced | expert>",
  "beginner_friendly": <true | false>,
  "hazards": ["<hazard1>", "<hazard2>"],
  "facilities": ["<facility1>", "<facility2>"],
  "best_swell_direction": ["<N>", "<NW>"],
  "best_swell_size_min_m": <float>,
  "best_swell_size_max_m": <float>,
  "best_swell_period_min": <float>,
  "best_wind_direction": ["<E>", "<NE>"],
  "best_tide": ["<low>", "<mid>", "<high>"],
  "best_season": ["<winter>", "<spring>", "<summer>", "<fall>"],
  "description": "<2-3 sentence description of the spot>",
  "local_tips": "<practical tips for visiting surfers>"
}}

Search results:
{search_results}

Respond with the JSON object only.
"""


class ResearchAgent(LoggerMixin):
    """Sub-agent that researches surf spots dynamically via web search.

    Workflow:
    1. Receives a spot query from the orchestrator.
    2. Runs a Tavily web search with surf-specific query enhancement.
    3. Passes search results to the LLM for structured extraction.
    4. Returns a structured dict the orchestrator can cache and pass to
       downstream tools (forecast, assessment, planning).
    """

    def __init__(self, llm_provider, settings):
        self._llm = llm_provider
        self._settings = settings

        # Lazy-init Tavily client
        self._tavily_client = None

    def _get_tavily_client(self):
        """Lazy-initialise the Tavily client."""
        if self._tavily_client is None:
            from tavily import TavilyClient

            api_key = self._settings.tavily.api_key
            if not api_key:
                raise RuntimeError(
                    "TAVILY_API_KEY is not configured. "
                    "Set it in your .env file (get a free key at https://tavily.com)."
                )
            self._tavily_client = TavilyClient(api_key=api_key)
        return self._tavily_client

    async def research_spot(self, query: str) -> dict[str, Any]:
        """Research a surf spot via web search + LLM extraction.

        Args:
            query: Free-text query, e.g. "Caparica" or "best surf spots in Bali".

        Returns:
            Structured dict with spot metadata (name, coordinates, break type, etc.)
            or an error dict if research fails.
        """
        self.log_info(f"Researching spot: {query}")

        # 1. Enhance query for surf-specific search
        search_query = f"{query} surf spot latitude longitude coordinates location break type hazards"

        # 2. Run Tavily search
        try:
            client = self._get_tavily_client()
            search_results = client.search(
                query=search_query,
                search_depth=self._settings.tavily.search_depth,
                max_results=self._settings.tavily.max_results,
            )
        except Exception as e:
            self.log_error(f"Tavily search failed: {e}")
            return {
                "error": f"Web search failed: {e}",
                "query": query,
                "suggestion": "Try providing more details (e.g., country or region).",
            }

        # 3. Format search results for LLM
        formatted_results = self._format_search_results(search_results)
        if not formatted_results:
            return {
                "error": f"No useful search results found for '{query}'.",
                "query": query,
                "suggestion": "Try a more specific query with the country or region name.",
            }

        # 4. Extract structured data via LLM
        extraction = self._extract_spot_info(formatted_results)

        # Apply known-coords override regardless of whether LLM extracted coordinates.
        # This prevents stochastic Tavily results from resolving the spot to the
        # wrong location when we have canonical coordinates on record.
        query_lower = query.lower()
        for token, (lat, lon) in _KNOWN_COORDS.items():
            if token in query_lower:
                if "error" in extraction:
                    extraction = extraction.get("partial_data", {})
                extraction["latitude"] = lat
                extraction["longitude"] = lon
                break

        if "error" in extraction:
            return extraction

        # 5. Add metadata
        extraction["_source"] = "research"
        extraction["_query"] = query
        self.log_info(f"Research complete for: {extraction.get('name', query)}")
        return extraction

    def _format_search_results(self, search_response: dict) -> str:
        """Format Tavily response into a clean text block for LLM extraction."""
        results = search_response.get("results", [])
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            parts.append(f"[{i}] {title}\n{content}\nSource: {url}\n")

        return "\n".join(parts)

    def _extract_spot_info(self, search_results_text: str) -> dict[str, Any]:
        """Use the LLM to extract structured spot info from search text."""
        prompt = EXTRACTION_PROMPT.format(search_results=search_results_text)

        try:
            response = self._llm.chat_with_tools(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
            )
            content = response.choices[0].message.content or ""

            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            # If LLM left lat/lon null, fall back to regex extraction from raw text.
            if not data.get("latitude") or not data.get("longitude"):
                lat_m = _LAT_RE.search(search_results_text)
                lon_m = _LON_RE.search(search_results_text)
                if lat_m:
                    data["latitude"] = float(lat_m.group(1))
                if lon_m:
                    data["longitude"] = float(lon_m.group(1))

            if not data.get("latitude") or not data.get("longitude"):
                return {
                    "error": "Could not determine spot coordinates from search results.",
                    "partial_data": data,
                    "suggestion": "Try a more specific query.",
                }

            return data

        except json.JSONDecodeError as e:
            self.log_error(f"LLM returned invalid JSON: {e}")
            return {
                "error": "Failed to parse spot information from search results.",
                "suggestion": "Try again with a more specific query.",
            }
        except Exception as e:
            self.log_error(f"LLM extraction failed: {e}")
            return {
                "error": f"Failed to extract spot information: {e}",
                "suggestion": "Try again or provide more details.",
            }

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return OpenAI function-calling schemas for research tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "research_spot",
                    "description": (
                        "Research a surf spot by searching the web for information "
                        "about it. Returns structured data including coordinates, "
                        "break type, hazards, skill levels, and best conditions. "
                        "Call this BEFORE fetch_forecast when the user mentions "
                        "any surf location."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The surf spot to research, e.g. 'Caparica' "
                                    "or 'best surf spots near Lisbon'"
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
