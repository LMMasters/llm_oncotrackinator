"""
Output generation for lesion tracking results.
"""

import json
from pathlib import Path
from typing import List, Union, Optional
from datetime import datetime

from llm_oncotrackinator.models import PatientLesionHistory


class OutputGenerator:
    """Generate output files from lesion tracking results."""

    @staticmethod
    def to_json(
        histories: Union[PatientLesionHistory, List[PatientLesionHistory]],
        file_path: Optional[str] = None,
        indent: int = 2
    ) -> str:
        """
        Convert lesion histories to JSON format.

        Args:
            histories: Single PatientLesionHistory or list of histories
            file_path: Optional path to save JSON file
            indent: JSON indentation level

        Returns:
            JSON string
        """
        if isinstance(histories, PatientLesionHistory):
            histories = [histories]

        # Convert to dict using Pydantic's model_dump
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_patients": len(histories),
            "patients": [history.model_dump(mode='json') for history in histories]
        }

        json_str = json.dumps(data, indent=indent)

        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(json_str)

        return json_str

    @staticmethod
    def to_summary(histories: List[PatientLesionHistory]) -> str:
        """
        Generate a human-readable summary of tracking results.

        Args:
            histories: List of patient lesion histories

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("LESION TRACKING SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total Patients: {len(histories)}")
        lines.append("")

        for history in histories:
            lines.append(f"Patient: {history.patient_id}")
            lines.append(f"  Timepoints: {len(history.timepoints)}")

            if history.timepoints:
                first_date = history.timepoints[0].date.date()
                last_date = history.timepoints[-1].date.date()
                lines.append(f"  Date Range: {first_date} to {last_date}")

            lesion_ids = history.get_all_lesion_ids()
            lines.append(f"  Unique Lesions: {len(lesion_ids)}")

            if lesion_ids:
                lines.append(f"  Lesion IDs: {', '.join(lesion_ids)}")

                # Show progression for each lesion
                for lesion_id in lesion_ids:
                    timeline = history.get_lesion_timeline(lesion_id)
                    if timeline:
                        lines.append(f"    {lesion_id} ({timeline[0].location}):")
                        for lesion in timeline:
                            size_str = ""
                            if lesion.size_cm:
                                size_str = f"{lesion.size_cm} cm"
                            elif lesion.size_mm:
                                size_str = f"{lesion.size_mm} mm"
                            date_str = lesion.timepoint_date.strftime("%Y-%m-%d")
                            lines.append(f"      - {date_str}: {size_str}")

            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    @staticmethod
    def save_summary(
        histories: List[PatientLesionHistory],
        file_path: str
    ) -> None:
        """
        Save a summary to a text file.

        Args:
            histories: List of patient lesion histories
            file_path: Path to save the summary file
        """
        summary = OutputGenerator.to_summary(histories)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(summary)
