"""
Test that all package components can be imported correctly.
"""

def test_imports():
    """Test all imports."""
    print("Testing package imports...")

    # Core components
    from llm_oncotrackinator import Config
    print("[OK] Config imported")

    from llm_oncotrackinator import DataLoader, MedicalReport
    print("[OK] DataLoader and MedicalReport imported")

    from llm_oncotrackinator import LesionExtractor
    print("[OK] LesionExtractor imported")

    from llm_oncotrackinator import LesionTracker
    print("[OK] LesionTracker imported")

    from llm_oncotrackinator import OutputGenerator
    print("[OK] OutputGenerator imported")

    from llm_oncotrackinator import Lesion, TimePoint, PatientLesionHistory
    print("[OK] Data models imported")

    # Test instantiation
    config = Config()
    print(f"[OK] Config created with model: {config.ollama_model}")

    loader = DataLoader(config=config)
    print("[OK] DataLoader created")

    extractor = LesionExtractor(config=config)
    print("[OK] LesionExtractor created")

    tracker = LesionTracker(config=config)
    print("[OK] LesionTracker created")

    print("\nAll imports and instantiations successful!")


if __name__ == "__main__":
    test_imports()
