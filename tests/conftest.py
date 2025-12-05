"""
Pytest configuration and shared fixtures.
"""

import pytest
from datetime import datetime

from llm_oncotrackinator import MedicalReport, Config


@pytest.fixture
def sample_config():
    """Provide a sample configuration."""
    return Config(
        ollama_model="llama3.1:8b",
        temperature=0.0,
        max_retries=2
    )


@pytest.fixture
def sample_report():
    """Provide a single sample medical report."""
    return MedicalReport(
        patient_id="P001",
        date=datetime(2024, 1, 15),
        report_text="CT scan of chest shows a 2.3 cm nodule in the right upper lobe."
    )


@pytest.fixture
def sample_reports():
    """Provide multiple sample medical reports for one patient."""
    return [
        MedicalReport(
            patient_id="P001",
            date=datetime(2024, 1, 15),
            report_text="CT scan of chest shows a 2.3 cm nodule in the right upper lobe. Small pleural effusion noted on the left side."
        ),
        MedicalReport(
            patient_id="P001",
            date=datetime(2024, 3, 20),
            report_text="Follow-up CT shows the right upper lobe nodule has increased to 2.8 cm. Left pleural effusion has resolved."
        ),
        MedicalReport(
            patient_id="P001",
            date=datetime(2024, 6, 10),
            report_text="CT demonstrates further growth of right upper lobe lesion to 3.2 cm. New 1.5 cm nodule identified in left lower lobe."
        )
    ]


@pytest.fixture
def multiple_patient_reports():
    """Provide reports for multiple patients."""
    return {
        "P001": [
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 1, 15),
                report_text="CT scan shows a 2.3 cm nodule in the right upper lobe."
            ),
            MedicalReport(
                patient_id="P001",
                date=datetime(2024, 3, 20),
                report_text="Follow-up shows the nodule has increased to 2.8 cm."
            )
        ],
        "P002": [
            MedicalReport(
                patient_id="P002",
                date=datetime(2024, 2, 1),
                report_text="MRI brain reveals a 1.2 cm enhancing lesion in the left frontal lobe."
            )
        ]
    }
