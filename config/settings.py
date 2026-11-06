"""
SurfSense Configuration Management

Centralized, type-safe configuration using Pydantic Settings.
All settings are loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAISettings(BaseSettings):
    """Configuration for Azure OpenAI Service."""

    model_config = SettingsConfigDict(env_prefix="AZURE_OPENAI_", env_file=".env", extra="ignore")

    endpoint: str = Field(
        default="",
        description="Azure OpenAI endpoint URL (e.g. https://<resource>.openai.azure.com/)",
    )
    api_key: str = Field(
        default="",
        description="Azure OpenAI API key",
    )
    deployment_name: str = Field(
        default="gpt-4o-mini",
        description="Azure deployment name for the model",
    )
    api_version: str = Field(
        default="2024-10-21",
        description="Azure OpenAI API version",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the orchestrator",
    )
    max_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens in LLM response",
    )


class LLMSettings(BaseSettings):
    """Configuration for the Language Model service."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = creative)",
    )
    max_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens in LLM response",
    )


class TavilySettings(BaseSettings):
    """Configuration for Tavily web search (used by ResearchAgent)."""

    model_config = SettingsConfigDict(env_prefix="TAVILY_", env_file=".env", extra="ignore")

    api_key: str = Field(
        default="",
        description="Tavily API key for web search",
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="Search depth: 'basic' (fast/cheap) or 'advanced' (thorough)",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search results to return",
    )


class ForecastAPISettings(BaseSettings):
    """Configuration for external forecast API."""

    model_config = SettingsConfigDict(env_prefix="FORECAST_", env_file=".env", extra="ignore")

    api_provider: str = Field(
        default="stormglass",
        description="Forecast data provider (surfline, stormglass)",
    )
    api_key: str = Field(
        default="",
        description="API key for forecast provider",
    )
    api_base_url: str = Field(
        default="https://api.stormglass.io/v2",
        description="Base URL for forecast API",
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="Cache time-to-live in seconds",
    )


class SkillLevelThresholds(BaseSettings):
    """Thresholds for surf condition suitability by skill level."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Beginner thresholds
    beginner_max_wave_height: float = Field(
        default=1.5,
        ge=0.5,
        description="Maximum wave height (meters) for beginners",
    )
    beginner_max_wind_speed: float = Field(
        default=15.0,
        ge=5.0,
        description="Maximum wind speed (km/h) for beginners",
    )

    # Intermediate thresholds
    intermediate_max_wave_height: float = Field(
        default=2.5,
        ge=1.0,
        description="Maximum wave height (meters) for intermediate surfers",
    )
    intermediate_max_wind_speed: float = Field(
        default=20.0,
        ge=10.0,
        description="Maximum wind speed (km/h) for intermediate surfers",
    )

    # Advanced thresholds
    advanced_max_wave_height: float = Field(
        default=5.0,
        ge=2.0,
        description="Maximum wave height (meters) for advanced surfers",
    )
    advanced_max_wind_speed: float = Field(
        default=30.0,
        ge=15.0,
        description="Maximum wind speed (km/h) for advanced surfers",
    )

    def get_thresholds(self, skill_level: str) -> dict[str, float]:
        """Get wave and wind thresholds for a given skill level."""
        thresholds = {
            "beginner": {
                "max_wave_height": self.beginner_max_wave_height,
                "max_wind_speed": self.beginner_max_wind_speed,
            },
            "intermediate": {
                "max_wave_height": self.intermediate_max_wave_height,
                "max_wind_speed": self.intermediate_max_wind_speed,
            },
            "advanced": {
                "max_wave_height": self.advanced_max_wave_height,
                "max_wind_speed": self.advanced_max_wind_speed,
            },
        }
        return thresholds.get(skill_level.lower(), thresholds["intermediate"])


class ScoringSettings(BaseSettings):
    """Configuration for the condition scoring mode (rule-based or ML)."""

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", populate_by_name=True
    )

    scoring_mode: Literal["rule", "ml"] = Field(
        default="ml",
        alias="SCORING_MODE",
        description="'rule' = deterministic heuristic, 'ml' = XGBoost model",
    )
    ml_model_path: str = Field(
        default="ml/models/surf_condition_model.joblib",
        alias="ML_MODEL_PATH",
        description="Path to the trained XGBoost model file",
    )


class LoggingSettings(BaseSettings):
    """Configuration for application logging."""

    model_config = SettingsConfigDict(env_prefix="LOG_", env_file=".env", extra="ignore")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level to record",
    )
    format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format",
    )
    file_path: Path = Field(
        default=Path("logs/surfsense.log"),
        description="Path to log file",
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_log_path(cls, v: str | Path) -> Path:
        """Ensure log directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    environment: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Application environment",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port",
    )
    session_timeout_minutes: int = Field(
        default=30,
        ge=5,
        description="Session timeout in minutes",
    )

    # Feature flags
    enable_local_forecast_fallback: bool = Field(
        default=True,
        description="Enable local model fallback when API fails",
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching",
    )
    enable_api_rate_limiting: bool = Field(
        default=True,
        description="Enable API rate limiting",
    )


class Settings(BaseSettings):
    """
    Root configuration container.

    Aggregates all configuration sections into a single, validated object.
    Use get_settings() to access the singleton instance.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI endpoint URL",
    )
    azure_openai_api_key: str = Field(
        default="",
        description="Azure OpenAI API key",
    )
    azure_openai_deployment_name: str = Field(
        default="",
        description="Azure OpenAI deployment name",
    )
    azure_openai_api_version: str = Field(
        default="2024-10-21",
        description="Azure OpenAI API version",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (used by LLM baseline evaluation)",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key (used by LLM baseline evaluation)",
    )

    # Nested configuration sections
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tavily: TavilySettings = Field(default_factory=TavilySettings)
    forecast: ForecastAPISettings = Field(default_factory=ForecastAPISettings)
    skill_thresholds: SkillLevelThresholds = Field(default_factory=SkillLevelThresholds)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    app: AppSettings = Field(default_factory=AppSettings)

    def validate_required_keys(self) -> list[str]:
        """
        Check that required Azure OpenAI configuration is set.

        Returns:
            List of missing required configuration keys.
        """
        missing = []

        if not self.azure_openai_api_key and not self.azure_openai.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.azure_openai_endpoint and not self.azure_openai.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.azure_openai_deployment_name and not self.azure_openai.deployment_name:
            missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Forecast API key is optional if local fallback is enabled
        if not self.forecast.api_key and not self.app.enable_local_forecast_fallback:
            missing.append("FORECAST_API_KEY")

        return missing

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings singleton.

    Settings are loaded once and cached for performance.
    Call this function to access configuration anywhere in the app.

    Returns:
        Settings: The validated application settings.

    Raises:
        ValidationError: If configuration values are invalid.
    """
    return Settings()


def validate_startup_config() -> None:
    """
    Validate configuration at application startup.

    Raises:
        SystemExit: If required configuration is missing.
    """
    settings = get_settings()
    missing_keys = settings.validate_required_keys()

    if missing_keys:
        error_message = (
            "\n🚫 Configuration Error: Missing required settings\n"
            f"   Missing: {', '.join(missing_keys)}\n"
            "   Please set these in your .env file or environment.\n"
            "   See .env.example for reference.\n"
        )
        raise SystemExit(error_message)
