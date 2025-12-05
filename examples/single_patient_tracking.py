"""
Simple example: Track lesions for a single patient.

NOTE: This requires Ollama to be running locally with a model installed.
"""

from datetime import datetime

from llm_oncotrackinator import (
    Config,
    MedicalReport,
    LesionTracker,
    OutputGenerator
)


def main():
    """Track lesions for a single patient with manually created reports."""

    print("Single Patient Lesion Tracking Example")
    print("=" * 60)
    print()

    # Configure
    config = Config(
        ollama_model="llama3.1:8b",
        temperature=0.0
    )

    # Create sample reports for one patient
    reports = [
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

    print(f"Tracking {len(reports)} timepoints for patient P001...")
    print()

    # Track lesions
    tracker = LesionTracker(config=config)
    history = tracker.track_patient("P001", reports)

    # Display results
    print(f"Patient: {history.patient_id}")
    print(f"Timepoints: {len(history.timepoints)}")
    print(f"Unique lesions tracked: {len(history.get_all_lesion_ids())}")
    print()

    # Show each timepoint
    for i, tp in enumerate(history.timepoints, 1):
        print(f"Timepoint {i} ({tp.date.date()}):")
        print(f"  Found {len(tp.lesions)} lesion(s)")
        for lesion in tp.lesions:
            size_str = f"{lesion.size_cm} cm" if lesion.size_cm else "size unknown"
            print(f"    - {lesion.lesion_id}: {lesion.location} ({size_str})")
        print()

    # Show lesion timelines
    print("Lesion Progression:")
    print("-" * 60)
    for lesion_id in history.get_all_lesion_ids():
        timeline = history.get_lesion_timeline(lesion_id)
        if timeline:
            print(f"{lesion_id} - {timeline[0].location}:")
            for obs in timeline:
                size_str = f"{obs.size_cm} cm" if obs.size_cm else "unknown"
                print(f"  {obs.timepoint_date.date()}: {size_str}")
            print()

    # Save as JSON
    json_output = OutputGenerator.to_json(history, indent=2)
    print("JSON Output:")
    print("-" * 60)
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)


if __name__ == "__main__":
    main()
