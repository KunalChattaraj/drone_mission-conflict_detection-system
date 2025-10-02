# debug_import.py
try:
    from conflict_detector import ConflictDetector
    print("✅ Successfully imported ConflictDetector")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
    # Show what's actually in the module
    import conflict_detector
    print("Available names in conflict_detector:")
    for name in dir(conflict_detector):
        if not name.startswith('_'):  # Skip private attributes
            print(f"  - {name}")