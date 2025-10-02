# create_test_missions.py
import pandas as pd

def create_test_mission_files():
    """Create all test mission CSV files"""
    
    # Safe Mission (far away)
    safe_data = {
        'x': [1000.0, 1100.0, 1200.0],
        'y': [1000.0, 1100.0, 1200.0],
        'z': [100.0, 110.0, 120.0],
        'start_time': ['2024-01-15 10:00:00'] * 3,
        'end_time': ['2024-01-15 10:30:00'] * 3
    }
    pd.DataFrame(safe_data).to_csv('safe_mission.csv', index=False)
    print("✅ Created safe_mission.csv")
    
    # Critical Conflict Mission (very close)
    critical_data = {
        'x': [5.0, 155.0, 305.0],
        'y': [5.0, 80.0, 155.0],
        'z': [102.0, 108.0, 118.0],
        'start_time': ['2024-01-15 10:05:00'] * 3,
        'end_time': ['2024-01-15 10:25:00'] * 3
    }
    pd.DataFrame(critical_data).to_csv('critical_conflict_mission.csv', index=False)
    print("✅ Created critical_conflict_mission.csv")
    
    # Moderate Conflict Mission (medium distance)
    moderate_data = {
        'x': [20.0, 170.0, 320.0],
        'y': [20.0, 95.0, 170.0],
        'z': [105.0, 112.0, 118.0],
        'start_time': ['2024-01-15 10:10:00'] * 3,
        'end_time': ['2024-01-15 10:20:00'] * 3
    }
    pd.DataFrame(moderate_data).to_csv('moderate_conflict_mission.csv', index=False)
    print("✅ Created moderate_conflict_mission.csv")
    
    # Different Time Mission
    time_data = {
        'x': [0.0, 150.0, 300.0],
        'y': [0.0, 75.0, 150.0],
        'z': [100.0, 110.0, 120.0],
        'start_time': ['2024-01-15 14:00:00'] * 3,
        'end_time': ['2024-01-15 14:30:00'] * 3
    }
    pd.DataFrame(time_data).to_csv('different_time_mission.csv', index=False)
    print("✅ Created different_time_mission.csv")
    
    # High Altitude Mission
    altitude_data = {
        'x': [0.0, 150.0, 300.0],
        'y': [0.0, 75.0, 150.0],
        'z': [200.0, 210.0, 220.0],
        'start_time': ['2024-01-15 10:00:00'] * 3,
        'end_time': ['2024-01-15 10:30:00'] * 3
    }
    pd.DataFrame(altitude_data).to_csv('high_altitude_mission.csv', index=False)
    print("✅ Created high_altitude_mission.csv")
    
    print("\n🎯 Test missions created! Use them with:")
    print("   python io.py")
    print("   Then choose option 2 (CSV file input)")

if __name__ == "__main__":
    create_test_mission_files()