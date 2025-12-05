"""
Basic usage example for LLM OncoTrackinator.
"""

from pathlib import Path
from llm_oncotrackinator import DataLoader, Config

# Configure the data loader
config = Config(
    patient_id_column="patient_id",
    date_column="date",
    report_column="report",
    ollama_model="llama3.1:8b"
)

# Initialize data loader
loader = DataLoader(config=config)

# Load data from CSV
data_path = Path(__file__).parent / "sample_data.csv"
reports = loader.load_csv(str(data_path))

print(f"Loaded {len(reports)} medical reports")
print(f"Number of unique patients: {len(set(r.patient_id for r in reports))}")

# Organize into patient timelines
timelines = loader.get_patient_timelines(reports)

# Display timeline summary
for patient_id, patient_reports in timelines.items():
    print(f"\n{patient_id}: {len(patient_reports)} reports from {patient_reports[0].date.date()} to {patient_reports[-1].date.date()}")
    for i, report in enumerate(patient_reports, 1):
        print(f"  Timepoint {i} ({report.date.date()}): {report.report_text[:80]}...")
