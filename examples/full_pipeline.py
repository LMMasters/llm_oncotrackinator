"""
Complete pipeline example for LLM OncoTrackinator.

This example demonstrates the full workflow:
1. Load medical reports from CSV
2. Extract and track lesions across timepoints
3. Generate structured JSON output
4. Create human-readable summary

NOTE: This requires Ollama to be running locally with a model installed.
"""

from pathlib import Path

from llm_oncotrackinator import (
    Config,
    DataLoader,
    LesionTracker,
    OutputGenerator
)


def main():
    """Run the complete lesion tracking pipeline."""

    print("=" * 80)
    print("LLM OncoTrackinator - Full Pipeline Example")
    print("=" * 80)
    print()

    # Step 1: Configure
    print("Step 1: Configuring...")
    config = Config(
        ollama_model="llama3.1:8b",  # Make sure this model is installed
        ollama_host="http://localhost:11434",
        temperature=0.0,  # Deterministic output
        max_retries=3
    )
    print(f"  Using model: {config.ollama_model}")
    print(f"  Temperature: {config.temperature}")
    print()

    # Step 2: Load data
    print("Step 2: Loading medical reports...")
    loader = DataLoader(config=config)
    data_path = Path(__file__).parent / "sample_data.csv"
    reports = loader.load_csv(str(data_path))
    print(f"  Loaded {len(reports)} reports")

    # Organize into patient timelines
    timelines = loader.get_patient_timelines(reports)
    print(f"  Found {len(timelines)} patients")
    print()

    # Step 3: Track lesions
    print("Step 3: Tracking lesions with LLM...")
    print("  (This may take a minute...)")
    tracker = LesionTracker(config=config)
    histories = tracker.track_all_patients(timelines)
    print(f"  Tracked lesions for {len(histories)} patients")
    print()

    # Step 4: Generate outputs
    print("Step 4: Generating outputs...")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save JSON output
    json_path = output_dir / "lesion_tracking_results.json"
    OutputGenerator.to_json(histories, file_path=str(json_path), indent=2)
    print(f"  Saved JSON: {json_path}")

    # Save summary
    summary_path = output_dir / "lesion_tracking_summary.txt"
    OutputGenerator.save_summary(histories, file_path=str(summary_path))
    print(f"  Saved summary: {summary_path}")
    print()

    # Step 5: Display summary
    print("Step 5: Results Summary")
    print("-" * 80)
    summary = OutputGenerator.to_summary(histories)
    print(summary)

    print()
    print("=" * 80)
    print("Pipeline completed successfully!")
    print(f"Check the outputs directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
