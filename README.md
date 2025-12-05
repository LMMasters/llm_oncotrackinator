# LLM OncoTrackinator

A professional Python package for tracking lesions in medical reports over time using Large Language Models (LLMs) via Ollama.

## Overview

LLM OncoTrackinator analyzes longitudinal medical reports (CT scans, MRIs, etc.) and automatically extracts and tracks lesions across multiple timepoints. The package uses local LLMs through Ollama to:

1. Extract lesion information (size, location) from the first timepoint
2. Track lesion changes over subsequent timepoints
3. Generate a structured JSON output with complete patient lesion histories

## Features

- **Professional data loading** with validation for CSV, Excel, and DataFrame inputs
- **Flexible configuration** for different dataset formats
- **Patient timeline management** for chronological report organization
- **Pydantic validation** for data integrity
- **Local LLM processing** via Ollama (privacy-focused, no cloud APIs)
- **Type-safe** with full type hints

## Installation

### Prerequisites

1. Python 3.8 or higher
2. [Ollama](https://ollama.ai/) installed and running locally

### Install the package

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-oncotrackinator.git
cd llm-oncotrackinator

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Set up Ollama

```bash
# Install Ollama from https://ollama.ai/

# Pull a model (e.g., llama3.1)
ollama pull llama3.1:8b
```

## Quick Start

### Prepare your data

Your dataset should be a CSV or Excel file with at least three columns:

- **patient_id**: Unique patient identifier
- **date**: Date of the report (any format pandas can parse)
- **report**: The medical report text

Example CSV:

```csv
patient_id,date,report
P001,2024-01-15,"CT scan shows a 2.3 cm nodule in the right upper lobe."
P001,2024-03-20,"Follow-up CT shows the nodule has increased to 2.8 cm."
```

### Complete Pipeline Example

```python
from llm_oncotrackinator import (
    Config,
    DataLoader,
    LesionTracker,
    OutputGenerator
)

# 1. Configure
config = Config(
    ollama_model="llama3.1:8b",
    temperature=0.0  # Deterministic output
)

# 2. Load medical reports
loader = DataLoader(config=config)
reports = loader.load_csv("medical_reports.csv")
timelines = loader.get_patient_timelines(reports)

# 3. Track lesions across timepoints (with progress bars!)
tracker = LesionTracker(config=config)
histories = tracker.track_all_patients(timelines)  # Shows progress by default

# 4. Generate outputs
OutputGenerator.to_json(histories, file_path="results.json")
OutputGenerator.save_summary(histories, file_path="summary.txt")

# 5. Access results programmatically
for history in histories:
    print(f"Patient {history.patient_id}")
    print(f"  Lesions tracked: {history.get_all_lesion_ids()}")

    # Get timeline for specific lesion
    for lesion_id in history.get_all_lesion_ids():
        timeline = history.get_lesion_timeline(lesion_id)
        for observation in timeline:
            print(f"  {lesion_id} at {observation.timepoint_date.date()}: {observation.size_cm} cm")
```

### Simple Single Patient Example

```python
from datetime import datetime
from llm_oncotrackinator import MedicalReport, LesionTracker, Config

# Create reports
reports = [
    MedicalReport(
        patient_id="P001",
        date=datetime(2024, 1, 15),
        report_text="CT scan shows a 2.3 cm nodule in the right upper lobe."
    ),
    MedicalReport(
        patient_id="P001",
        date=datetime(2024, 3, 20),
        report_text="Follow-up CT shows the nodule has increased to 2.8 cm."
    )
]

# Track lesions
tracker = LesionTracker(config=Config())
history = tracker.track_patient("P001", reports)

# View results
print(f"Tracked {len(history.get_all_lesion_ids())} lesions across {len(history.timepoints)} timepoints")
```

## Project Structure

```plaintext
llm-oncotrackinator/
├── src/
│   └── llm_oncotrackinator/
│       ├── __init__.py           # Package exports
│       ├── config.py             # Configuration management
│       ├── data_loader.py        # Data loading and validation
│       ├── models.py             # Pydantic data models
│       ├── lesion_extractor.py   # LLM-based lesion extraction
│       ├── tracker.py            # Lesion tracking across timepoints
│       └── output.py             # JSON and summary generation
├── examples/
│   ├── sample_data.csv           # Example dataset
│   ├── basic_usage.py            # Basic data loading example
│   ├── single_patient_tracking.py # Single patient example
│   └── full_pipeline.py          # Complete pipeline example
├── tests/
│   └── test_data_loader.py       # Unit tests
├── outputs/                      # Generated results (gitignored)
├── pyproject.toml                # Package configuration
├── requirements.txt              # Pip dependencies
├── environment.yml               # Conda environment
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## Configuration Options

The `Config` class supports the following parameters:

```python
Config(
    ollama_model="llama3.1:8b",              # Ollama model to use
    ollama_host="http://localhost:11434",    # Ollama server URL
    patient_id_column="patient_id",          # Column name for patient IDs
    date_column="date",                      # Column name for dates
    report_column="report",                  # Column name for report text
    temperature=0.0,                         # LLM temperature (0=deterministic)
    max_retries=3                            # Max retries for API calls
)
```

## Data Requirements

### Required Columns

- Patient identifier (default: `patient_id`)
- Report date (default: `date`)
- Report text (default: `report`)

### Data Quality

- Patient IDs and report text cannot be empty
- Dates must be parseable by pandas
- Rows with missing required data are automatically dropped with a warning

### Supported Formats

- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)
- Pandas DataFrames

## Features

- [x] Data loading and validation
- [x] Configuration management
- [x] LLM-based lesion extraction
- [x] Temporal lesion tracking across timepoints
- [x] JSON output generation
- [x] Human-readable summary reports
- [x] Patient timeline organization
- [x] Pydantic v2 data validation
- [ ] Comprehensive unit tests
- [ ] Advanced prompting strategies
- [ ] Support for custom extraction schemas
- [ ] Batch processing optimization

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{llm_oncotrackinator,
  title={LLM OncoTrackinator: Automated Lesion Tracking in Medical Reports},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm-oncotrackinator}
}
```

## Disclaimer

This software is for research purposes only and is not intended for clinical use. Always verify results with qualified healthcare professionals.
