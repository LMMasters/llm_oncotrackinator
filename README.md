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

### Load and process data

```python
from llm_oncotrackinator import DataLoader, Config

# Configure the loader
config = Config(
    patient_id_column="patient_id",
    date_column="date",
    report_column="report",
    ollama_model="llama3.1:8b"
)

# Load data
loader = DataLoader(config=config)
reports = loader.load_csv("medical_reports.csv")

# Organize into patient timelines
timelines = loader.get_patient_timelines(reports)

print(f"Loaded {len(reports)} reports for {len(timelines)} patients")
```

## Project Structure

```
llm-oncotrackinator/
├── src/
│   └── llm_oncotrackinator/
│       ├── __init__.py          # Package exports
│       ├── config.py             # Configuration management
│       ├── data_loader.py        # Data loading and validation
│       ├── lesion_extractor.py   # LLM-based lesion extraction (coming soon)
│       └── tracker.py            # Lesion tracking logic (coming soon)
├── examples/
│   ├── sample_data.csv           # Example dataset
│   └── basic_usage.py            # Usage examples
├── tests/                        # Unit tests (coming soon)
├── pyproject.toml                # Package configuration
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

## Roadmap

- [x] Data loading and validation
- [x] Configuration management
- [ ] LLM-based lesion extraction
- [ ] Temporal lesion tracking
- [ ] JSON output generation
- [ ] Comprehensive tests
- [ ] Advanced prompting strategies
- [ ] Support for custom extraction schemas

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
