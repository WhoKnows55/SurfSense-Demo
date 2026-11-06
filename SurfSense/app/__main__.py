"""
Main entry point for running the SurfSense application.

Run with: python -m app

Architecture:
- Layer 1: ConversationalAgent - user-facing dialogue management
- Layer 2: Contextual Layer - parking, accessibility, reviews, safety
- Layer 3: ForecastIntegrationAgent - forecast API integration
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import __version__
from app.core.logger import get_logger, setup_logging
from config.settings import get_settings, validate_startup_config


def print_banner() -> None:
    """Display the SurfSense welcome banner."""
    print("=" * 60)
    print("🏄 SurfSense - AI Surf Trip Planning Assistant")
    print("=" * 60)
    print("   Powered by Azure OpenAI orchestrator with sub-agents:")
    print("   • Orchestrator (GPT-4o-mini, dialogue & function-calling)")
    print("   • Research Agent (Tavily web search)")
    print("   • Forecast Data Agent (surf conditions)")
    print("   • Condition Assessment Agent (skill-based evaluation)")
    print("   • Trip Planning Agent (itinerary optimisation)")


def print_config_summary(settings) -> None:
    """Display current configuration summary."""
    print(f"\n📋 Configuration Summary:")
    if settings.azure_openai.endpoint and settings.azure_openai.api_key:
        print(f"   Provider: Azure OpenAI ({settings.azure_openai.deployment_name})")
    elif getattr(settings, "openai_api_key", ""):
        print(f"   Provider: OpenAI ({settings.azure_openai.deployment_name})")
    print(f"   Temperature: {settings.azure_openai.temperature}")


async def chat_loop(settings) -> None:
    """
    Run the interactive chat loop using the orchestrator architecture.

    Args:
        settings: Application settings.
    """
    from app.agents.orchestrator import Orchestrator
    from app.core.llm_service import AzureOpenAIProvider, OpenAILLMProvider

    print("\n🏄 Initialising SurfSense...")

    # Initialise LLM provider (Azure OpenAI or standard OpenAI)
    try:
        openai_key = (getattr(settings, "openai_api_key", "") or "").strip()
        azure_key = (settings.azure_openai.api_key or "").strip()
        azure_endpoint = (settings.azure_openai.endpoint or "").strip()
        if openai_key:
            llm_provider = OpenAILLMProvider(
                api_key=openai_key,
                model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
                max_tokens=settings.azure_openai.max_tokens,
                temperature=settings.azure_openai.temperature,
            )
        elif azure_endpoint and azure_key:
            llm_provider = AzureOpenAIProvider(
                endpoint=azure_endpoint,
                api_key=azure_key,
                deployment_name=settings.azure_openai.deployment_name,
                api_version=settings.azure_openai.api_version,
                temperature=settings.azure_openai.temperature,
                max_tokens=settings.azure_openai.max_tokens,
            )
        else:
            print("\n❌ No API key configured. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY in .env")
            return
    except Exception as e:
        print(f"\n❌ Failed to initialise LLM provider: {e}")
        return

    # Initialise orchestrator
    orchestrator = Orchestrator(llm_provider, settings)

    print("\n✅ Ready! Type 'quit' or 'exit' to leave.")
    print("   Commands: /reset, /help")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        # Handle slash commands
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "reset":
                orchestrator.reset()
                print("\n🔄 Conversation reset.")
            elif cmd == "help":
                print("\n📖 Available commands:")
                print("   /reset - Reset conversation")
                print("   /help  - Show this help")
                print("\n   Everything else is handled through natural conversation.")
                print("   Just ask about forecasts, conditions, trip planning, etc.")
            else:
                print(f"\n❓ Unknown command: /{cmd}")
                print("   Type /help for available commands.")
            continue

        # Process through orchestrator
        try:
            print("\n🤖 SurfSense: ", end="", flush=True)
            response = await orchestrator.process(user_input)
            print(response)
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main() -> None:
    """Main application entry point."""
    print_banner()
    print(f"\nVersion: {__version__}")

    # Load and validate configuration
    try:
        settings = get_settings()

        # Setup logging (quieter for terminal chat)
        setup_logging(
            level="WARNING",  # Suppress info logs in chat mode
            log_format="text",
            log_file=settings.logging.file_path,
        )

        logger = get_logger(__name__)
        logger.info("SurfSense starting up", extra={"version": __version__})

        # Validate required keys (warning only in development)
        missing_keys = settings.validate_required_keys()
        if missing_keys:
            if settings.is_production():
                validate_startup_config()
            else:
                print(f"\n⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
                print("   Some features may not work.")

        print_config_summary(settings)

        # Start chat loop with async support
        asyncio.run(chat_loop(settings))

    except SystemExit:
        raise
    except Exception as e:
        print(f"\n❌ Startup Error: {e}")
        print("   Check your .env file and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
