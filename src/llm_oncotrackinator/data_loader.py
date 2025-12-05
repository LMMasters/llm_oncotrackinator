"""
Data loading and validation for medical reports.
"""

from typing import List, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_oncotrackinator.config import Config


class MedicalReport(BaseModel):
    """A single medical report entry."""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    patient_id: str = Field(..., description="Unique patient identifier")
    date: datetime = Field(..., description="Date of the report")
    report_text: str = Field(..., description="The medical report text content")

    @field_validator("report_text")
    @classmethod
    def validate_report_not_empty(cls, v: str) -> str:
        """Ensure report text is not empty."""
        if not v or not v.strip():
            raise ValueError("Report text cannot be empty")
        return v.strip()

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, v: str) -> str:
        """Ensure patient ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Patient ID cannot be empty")
        return v.strip()


class DataLoader:
    """Load and validate medical report datasets."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the DataLoader.

        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Config()

    def load_csv(self, file_path: str) -> List[MedicalReport]:
        """
        Load medical reports from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of validated MedicalReport objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing or data is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        return self._load_from_dataframe(df)

    def load_excel(self, file_path: str, sheet_name: Optional[str] = None) -> List[MedicalReport]:
        """
        Load medical reports from an Excel file.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to read. If None, reads the first sheet.

        Returns:
            List of validated MedicalReport objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing or data is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return self._load_from_dataframe(df)

    def load_from_dataframe(self, df: pd.DataFrame) -> List[MedicalReport]:
        """
        Load medical reports from a pandas DataFrame.

        Args:
            df: DataFrame containing medical reports

        Returns:
            List of validated MedicalReport objects

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        return self._load_from_dataframe(df)

    def _load_from_dataframe(self, df: pd.DataFrame) -> List[MedicalReport]:
        """
        Internal method to load and validate medical reports from a DataFrame.

        Args:
            df: DataFrame containing medical reports

        Returns:
            List of validated MedicalReport objects sorted by patient_id and date

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate required columns exist
        required_columns = {
            self.config.patient_id_column,
            self.config.date_column,
            self.config.report_column
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

        # Parse dates
        df = df.copy()
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])

        # Remove rows with missing required data
        initial_count = len(df)
        df = df.dropna(subset=[
            self.config.patient_id_column,
            self.config.date_column,
            self.config.report_column
        ])
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            print(f"Warning: Dropped {dropped_count} rows with missing required data")

        # Sort by patient_id and date
        df = df.sort_values([self.config.patient_id_column, self.config.date_column])

        # Convert to MedicalReport objects
        reports = []
        errors = []

        for idx, row in df.iterrows():
            try:
                report = MedicalReport(
                    patient_id=str(row[self.config.patient_id_column]),
                    date=row[self.config.date_column],
                    report_text=str(row[self.config.report_column])
                )
                reports.append(report)
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")

        if errors:
            error_summary = "\n".join(errors[:10])  # Show first 10 errors
            if len(errors) > 10:
                error_summary += f"\n... and {len(errors) - 10} more errors"
            raise ValueError(f"Data validation failed:\n{error_summary}")

        if not reports:
            raise ValueError("No valid reports found in the dataset")

        return reports

    def get_patient_timelines(
        self,
        reports: List[MedicalReport]
    ) -> dict[str, List[MedicalReport]]:
        """
        Organize reports into patient timelines.

        Args:
            reports: List of medical reports

        Returns:
            Dictionary mapping patient_id to chronologically sorted list of reports
        """
        timelines: dict[str, List[MedicalReport]] = {}

        for report in reports:
            if report.patient_id not in timelines:
                timelines[report.patient_id] = []
            timelines[report.patient_id].append(report)

        # Ensure each timeline is sorted by date
        for patient_id in timelines:
            timelines[patient_id].sort(key=lambda r: r.date)

        return timelines
