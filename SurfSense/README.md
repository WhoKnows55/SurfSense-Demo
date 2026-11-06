---
title: SurfSense
emoji: 🏄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
app_file: app_gradio.py
pinned: false
---

# 🏄 SurfSense - AI Surf Trip Planning Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A **terminal-based conversational AI assistant** that helps surfers plan trips by analyzing surf forecasts, evaluating conditions against skill levels, and creating optimized multi-day itineraries. Powered by **Azure OpenAI GPT-4o-mini** with function-calling, delegating to deterministic sub-agents for data aggregation, condition assessment, and trip planning.

## 🌊 Features

- **🤖 Azure OpenAI Orchestrator**: GPT-4o-mini with function-calling manages dialogue and delegates to specialized sub-agents
- **🔍 Dynamic Spot Research**: Tavily-powered web search + LLM extraction — ask about *any* surf spot worldwide, no pre-built database needed
- **💬 Terminal Chat Interface**: Natural conversation — no slash commands needed, just describe your trip
- **🌊 Multi-Source Forecasts**: Integrates Stormglass (paid) and Open-Meteo (free, no API key) for wave, swell, wind, and tide data
- **📊 Skill-Level Safety**: Deterministic scoring evaluates conditions against beginner/intermediate/advanced thresholds
- **📅 Trip Optimization**: Greedy multi-day itinerary planning with travel-time penalties (Haversine) and spot diversity
- **🏖️ Contextual Data**: Parking, accessibility, reviews, and safety information for surf spots

## 📋 Prerequisites

- **Python 3.10+**
- **Azure OpenAI** API access (GPT-4o-mini deployment with function-calling support)
- **Tavily** API key (free tier: 1,000 searches/month — get one at [tavily.com](https://tavily.com))
- macOS, Linux, or Windows

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd SurfSense

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
make install
# Or manually:
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your credentials:
#   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
#   AZURE_OPENAI_API_KEY=<your-key>
#   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
#   TAVILY_API_KEY=tvly-<your-key>
```

### 4. Verify API Connections

```bash
make check-api
# Or:
pytest tests/test_api_connections.py -v
```

This pings every external service (Azure OpenAI / OpenAI, Tavily, Stormglass, Open-Meteo, NOAA) and confirms your API keys are valid. Fix any failures before starting the app.

### 5. Start Chatting!

```bash
make run
# Or:
python -m app
```

## 💬 Example Conversation

```
============================================================
🏄 SurfSense - AI Surf Trip Planning Assistant
============================================================

Version: 0.1.0

📋 Configuration Summary:
   Azure OpenAI Deployment: gpt-4o

✅ Ready! Type 'quit' or 'exit' to leave.

------------------------------------------------------------

🧑 You: How's Caparica looking this weekend? I'm an intermediate surfer.

🤖 SurfSense: Let me research that spot and check the forecast...

   📍 Costa da Caparica, Portugal (beach break, sand bottom)
   Skill level: Beginner–Intermediate friendly

   Saturday: 3-4ft waves, light E wind (offshore), 12s period
   Rating: ✅ Ideal for intermediate — clean conditions all morning

   Sunday: 2-3ft, NW wind picking up by noon
   Rating: ⚠️ Suitable until 11am, then conditions deteriorate

   Best window: Saturday 7am–12pm (score: 82/100)

   Hazards: rip currents near the northern end
   Parking: free lots along the coast road

🧑 You: quit

👋 Goodbye!
```

## 📁 Project Structure

```
SurfSense/
├── app/
│   ├── __init__.py              # Package info and version
│   ├── __main__.py              # Terminal chat entry point
│   ├── agents/
│   │   ├── orchestrator.py      # LLM-powered orchestrator (function-calling)
│   │   ├── research_agent.py    # Dynamic spot research (Tavily + LLM)
│   │   ├── forecast_data_agent.py   # Data aggregation sub-agent
│   │   ├── condition_agent.py       # Condition assessment sub-agent
│   │   └── trip_planning_agent.py   # Trip planning sub-agent
│   ├── core/
│   │   ├── llm_service.py       # Azure OpenAI + fallback providers
│   │   └── logger.py            # Structured logging with redaction
│   ├── forecasting/
│   │   ├── models.py            # Unified forecast data models (Pydantic)
│   │   ├── openmeteo_client.py  # Open-Meteo API (free, no key)
│   │   ├── stormglass_client.py # Stormglass API (paid, 10 req/day free)
│   │   └── noaa_client.py       # NOAA Marine Weather (US only)
│   ├── contextual/
│   │   ├── parking.py           # Parking data provider
│   │   ├── accessibility.py     # Accessibility data provider
│   │   ├── reviews.py           # Reviews data provider
│   │   └── safety.py            # Safety/hazard data provider
│   ├── planning/
│   │   ├── condition_assessor.py    # Deterministic condition scoring
│   │   ├── window_finder.py         # Surf window identification
│   │   ├── trip_planner.py          # Multi-day itinerary optimization
│   │   └── travel_utils.py          # Haversine distance calculations
│   └── knowledge/
│       └── spot_database.py     # Legacy spot database (deprecated)
├── config/
│   └── settings.py              # Type-safe Pydantic configuration
├── data/
│   └── spots.json               # Legacy spot metadata (deprecated)
├── tests/                       # Unit tests for sub-agents
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
├── Makefile                     # Common commands
└── README.md
```

## ⚙️ Configuration

All settings are in `.env` (copy from `.env.example`):

### Azure OpenAI (Required)

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | *(empty)* | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | *(empty)* | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | `gpt-4o` | Azure deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | API version |
| `AZURE_OPENAI_TEMPERATURE` | `0.7` | Sampling temperature (0.0–2.0) |
| `AZURE_OPENAI_MAX_TOKENS` | `2000` | Max tokens in LLM response |

### Tavily Search (Required)

| Variable | Default | Description |
|----------|---------|-------------|
| `TAVILY_API_KEY` | *(empty)* | Tavily API key ([get free key](https://tavily.com)) |
| `TAVILY_SEARCH_DEPTH` | `basic` | Search depth: `basic` (fast) or `advanced` (thorough) |
| `TAVILY_MAX_RESULTS` | `5` | Number of search results per query (1–10) |

### Forecast API (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_API_KEY` | *(empty)* | Stormglass API key. Open-Meteo is used as free fallback |

### Skill Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `BEGINNER_MAX_WAVE_HEIGHT` | `1.5` | Max wave height (m) for beginner |
| `BEGINNER_MAX_WIND_SPEED` | `15` | Max wind speed (kph) for beginner |
| `INTERMEDIATE_MAX_WAVE_HEIGHT` | `2.5` | Max wave height (m) for intermediate |
| `INTERMEDIATE_MAX_WIND_SPEED` | `20` | Max wind speed (kph) for intermediate |
| `ADVANCED_MAX_WAVE_HEIGHT` | `5.0` | Max wave height (m) for advanced |
| `ADVANCED_MAX_WIND_SPEED` | `30` | Max wind speed (kph) for advanced |

## 🛠️ Development

### Make Commands

```bash
make help      # Show all available commands
make install   # Install dependencies
make run       # Start the chat application
make test      # Run test suite
make check-api # Verify all API connections
make clean     # Remove cache files
```

### Chat Commands

| Command | Description |
|---------|-------------|
| `/reset` | Clear conversation history and cached data |
| `/help` | Show available commands |
| `quit` / `exit` | Exit the application |

## 🏗️ Architecture

SurfSense follows a **single-orchestrator, multi-agent** pattern:

```
User (Terminal)
      │
      ▼
┌──────────────────────────────────────────────────┐
│          Orchestrator (LLM-powered)              │
│  Azure OpenAI GPT-4o-mini with function-calling       │
│  • Manages dialogue and preference elicitation   │
│  • Selects which sub-agent tool to call          │
│  • Synthesises sub-agent outputs into responses  │
└──┬──────────┬──────────────┬──────────────┬──────┘
   │          │              │              │
   ▼          ▼              ▼              ▼
┌────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│Research│ │  Forecast & │ │  Condition   │ │    Trip      │
│ Agent  │ │    Data     │ │  Assessment  │ │  Planning    │
│        │ │ Aggregation │ │    Agent     │ │    Agent     │
│research│ │    Agent    │ │              │ │              │
│_spot   │ │             │ │ • assess_    │ │ • find_surf_ │
│        │ │ • fetch_    │ │   conditions │ │   windows    │
│Tavily +│ │   forecast  │ │ • check_    │ │ • plan_      │
│LLM     │ │ • fetch_    │ │   safety    │ │   itinerary  │
│extract │ │   context   │ │ • get_skill_│ │ • rank_spots │
│        │ │ • get_spot_ │ │   thresholds│ │              │
│        │ │   metadata  │ │              │ │              │
└────────┘ └─────────────┘ └──────────────┘ └──────────────┘
   │              │                │                  │
   ▼              ▼                ▼                  ▼
 Tavily       External APIs    config/settings.py   Haversine +
 Web Search   (Open-Meteo,     (SkillLevel          greedy
              Stormglass)      Thresholds)         optimisation
```

### Data Flow

When a user mentions any surf spot, the orchestrator follows this sequence:

1. **`research_spot`** → Tavily web search + LLM extraction → structured spot data (coordinates, break type, hazards, skill levels)
2. **`fetch_forecast`** → uses coordinates from step 1 → hourly wave/wind/swell data
3. **`assess_conditions`** → scores each hour against the user's skill level
4. **`find_surf_windows`** / **`plan_itinerary`** → identifies best times and builds schedules

### Design Principles

- **Single LLM point**: Only the orchestrator calls Azure OpenAI — predictable token costs, no non-determinism in safety scoring
- **Dynamic knowledge**: No hardcoded spot database — the ResearchAgent discovers spot information at conversation time via web search
- **Function-calling as delegation**: GPT-4o-mini decides which tools to invoke; tool results feed back into the conversation
- **Deterministic sub-agents**: Python classes with scoring formulas, API calls, and optimization algorithms — no LLM calls (except ResearchAgent's extraction step)

## 🧪 Testing

```bash
# Run all tests
make test

# Or with pytest directly
pytest tests/ -v
```

### Verify API Connections

Before your first run (or after changing `.env`), check that every external
service is reachable and your keys are valid:

```bash
make check-api
```

This runs one lightweight ping per API:

| Test | Service | Key required? |
|------|---------|:---:|
| `TestLLMConnection` | Azure OpenAI / OpenAI | ✅ |
| `TestTavilyConnection` | Tavily web search | ✅ |
| `TestStormglassConnection` | Stormglass forecast | Optional |
| `TestOpenMeteoConnection` | Open-Meteo forecast | ❌ Free |
| `TestNOAAConnection` | NOAA weather | ❌ Free |

## 🆘 Troubleshooting

### Azure OpenAI Connection Error?

Run `make check-api` to pinpoint which service is failing.

Verify your credentials in `.env`:

```bash
# Test your Azure OpenAI setup
python -c "from openai import AzureOpenAI; print('OK')"
```

Ensure your deployment supports function-calling (GPT-4o-mini recommended).

### Import Errors?

```bash
# Ensure venv is activated and dependencies installed
source .venv/bin/activate
pip install -r requirements.txt
```

### Forecast Data Unavailable?

Open-Meteo (free, no API key) is used as the default forecast source. If it's down, configure Stormglass as a fallback:

```bash
# In .env:
FORECAST_API_KEY=<your-stormglass-key>
```

---

**Built with ❤️ for the surfing community** 🌊