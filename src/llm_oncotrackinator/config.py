"""
Configuration management for LLM OncoTrackinator.
"""

from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    """Configuration for the LLM OncoTrackinator."""

    model_config = ConfigDict(validate_assignment=True)

    ollama_model: str = Field(
        default="llama3.1:8b",
        description="The Ollama model to use for lesion extraction"
    )

    ollama_host: str = Field(
        default="http://localhost:11434",
        description="The Ollama server host URL"
    )

    patient_id_column: str = Field(
        default="patient_id",
        description="Name of the patient ID column in the dataset"
    )

    date_column: str = Field(
        default="date",
        description="Name of the date column in the dataset"
    )

    report_column: str = Field(
        default="report",
        description="Name of the report text column in the dataset"
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation (0 = deterministic)"
    )

    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for LLM API calls"
    )
