import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

def create_sample_missions():
    """
    Create sample mission data with mission_id, waypoints, t_start, and t_end.
    Returns a dictionary of missions.
    """
    
    # Base time for all missions
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    missions = {
        "mission_001": {
            "mission_id": "mission_001",
            "waypoints": [
                [0.0, 0.0, 100.0],    # Start point
                [100.0, 50.0, 110.0], # Intermediate
                [200.0, 100.0, 120.0], # Intermediate  
                [300.0, 150.0, 110.0], # Intermediate
                [400.0, 200.0, 100.0]  # End point
            ],
            "t_start": base_time,
            "t_end": base_time + timedelta(minutes=30)
        },
        "mission_002": {
            "mission_id": "mission_002", 
            "waypoints": [
                [50.0, -50.0, 90.0],
                [150.0, 0.0, 95.0],
                [250.0, 50.0, 100.0],
                [350.0, 100.0, 95.0]
            ],
            "t_start": base_time + timedelta(minutes=10),
            "t_end": base_time + timedelta(minutes=35)
        },
        "mission_003": {
            "mission_id": "mission_003",
            "waypoints": [
                [-100.0, 100.0, 120.0],
                [0.0, 150.0, 125.0], 
                [100.0, 200.0, 130.0],
                [200.0, 250.0, 125.0],
                [300.0, 300.0, 120.0]
            ],
            "t_start": base_time + timedelta(minutes=5),
            "t_end": base_time + timedelta(minutes=40)
        },
        "mission_004": {
            "mission_id": "mission_004",
            "waypoints": [
                [400.0, 0.0, 105.0],  # Far away - should not conflict
                [500.0, -100.0, 110.0],
                [600.0, -200.0, 105.0]
            ],
            "t_start": base_time + timedelta(minutes=15),
            "t_end": base_time + timedelta(minutes=45)
        },
        "mission_005": {
            "mission_id": "mission_005",
            "waypoints": [
                [0.0, 10.0, 102.0],   # Very close to mission_001 start (2m vertical separation)
                [150.0, 60.0, 108.0], # Close to mission_001 intermediate
                [280.0, 140.0, 112.0] # Close to mission_001 intermediate
            ],
            "t_start": base_time + timedelta(minutes=5),
            "t_end": base_time + timedelta(minutes=25)
        }
    }
    
    return missions

def convert_to_dataframe(mission_dict: dict) -> pd.DataFrame:
    """
    Convert a mission dictionary to the DataFrame format expected by the deconfliction system.
    """
    waypoints = mission_dict["waypoints"]
    df = pd.DataFrame(waypoints, columns=["x", "y", "z"])
    
    # Add start_time and end_time columns (same for all rows as per your system)
    df["start_time"] = pd.to_datetime(mission_dict["t_start"]).tz_localize("UTC")
    df["end_time"] = pd.to_datetime(mission_dict["t_end"]).tz_localize("UTC")
    
    return df

def export_missions_to_csv(missions_dict, export_dir=None, export_both_formats=True):
    """
    Optional feature: Export missions to CSV files
    
    Parameters:
    - missions_dict: Dictionary containing mission data
    - export_dir: Directory to export files (None for current directory)
    - export_both_formats: If True, exports both compact and expanded formats
    """
    if export_dir is None:
        export_dir = os.getcwd()
    
    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    export_info = {}
    
    if export_both_formats:
        # Export compact format (one row per mission)
        compact_records = []
        for mission_id, mission_data in missions_dict.items():
            record = {
                "mission_id": mission_data["mission_id"],
                "start_time": mission_data["t_start"],
                "end_time": mission_data["t_end"],
                "waypoints": str(mission_data["waypoints"]),
                "waypoint_count": len(mission_data["waypoints"])
            }
            compact_records.append(record)
        
        compact_df = pd.DataFrame(compact_records)
        compact_df["start_time"] = pd.to_datetime(compact_df["start_time"])
        compact_df["end_time"] = pd.to_datetime(compact_df["end_time"])
        
        compact_path = os.path.join(export_dir, "missions_compact.csv")
        compact_df.to_csv(compact_path, index=False)
        export_info["compact"] = compact_path
        print(f"✅ Compact format exported: {compact_path}")
        
        # Export expanded format (one row per waypoint)
        expanded_records = []
        for mission_id, mission_data in missions_dict.items():
            for i, waypoint in enumerate(mission_data["waypoints"]):
                record = {
                    "mission_id": mission_data["mission_id"],
                    "waypoint_index": i,
                    "x": waypoint[0],
                    "y": waypoint[1],
                    "z": waypoint[2],
                    "start_time": mission_data["t_start"],
                    "end_time": mission_data["t_end"]
                }
                expanded_records.append(record)
        
        expanded_df = pd.DataFrame(expanded_records)
        expanded_df["start_time"] = pd.to_datetime(expanded_df["start_time"])
        expanded_df["end_time"] = pd.to_datetime(expanded_df["end_time"])
        
        expanded_path = os.path.join(export_dir, "missions_expanded.csv")
        expanded_df.to_csv(expanded_path, index=False)
        export_info["expanded"] = expanded_path
        print(f"✅ Expanded format exported: {expanded_path}")
    
    return export_info

# =============================================================================
# MAIN EXECUTION WITH OPTIONAL CSV EXPORT
# =============================================================================

# Create sample missions
sample_missions = create_sample_missions()

# Demonstrate conversion to DataFrame format
print("Sample Mission Structure:")
print("=" * 50)
for mission_id, mission_data in sample_missions.items():
    print(f"\nMission ID: {mission_data['mission_id']}")
    print(f"Time: {mission_data['t_start'].strftime('%H:%M')} to {mission_data['t_end'].strftime('%H:%M')}")
    print(f"Waypoints: {len(mission_data['waypoints'])} points")
    print(f"First waypoint: {mission_data['waypoints'][0]}")
    print(f"Last waypoint: {mission_data['waypoints'][-1]}")
    
    # Convert to DataFrame to show the format
    df = convert_to_dataframe(mission_data)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Start time: {df['start_time'].iloc[0]}")
    print(f"End time: {df['end_time'].iloc[0]}")
    print("-" * 30)

# Create a dictionary of DataFrames for the deconfliction system
simulated_flights = {}
for mission_id, mission_data in sample_missions.items():
    if mission_id != "mission_001":  # Use mission_001 as primary, others as simulated
        simulated_flights[mission_id] = convert_to_dataframe(mission_data)

# # Show the primary mission
# primary_mission = convert_to_dataframe(sample_missions["mission_001"])
# print(f"\nPrimary Mission DataFrame:")
# print(primary_mission)
# print(f"\nPrimary mission times: {primary_mission['start_time'].iloc[0]} to {primary_mission['end_time'].iloc[0]}")

# print(f"\nSimulated flights: {list(simulated_flights.keys())}")

# =============================================================================
# OPTIONAL CSV EXPORT FEATURE
# =============================================================================
print("\n" + "="*60)
print("OPTIONAL CSV EXPORT")
print("="*60)

# Ask user if they want to export to CSV
export_csv = input("Do you want to export missions to CSV files? (y/n): ").lower().strip()

if export_csv in ['y', 'yes', '1']:
    # Get export directory (optional)
    export_dir = input("Enter export directory (press Enter for current directory): ").strip()
    if not export_dir:
        export_dir = None
    
    # Export missions
    try:
        exported_files = export_missions_to_csv(sample_missions, export_dir)
        
        print(f"\n📊 Export Summary:")
        print(f"   Total missions: {len(sample_missions)}")
        total_waypoints = sum(len(mission["waypoints"]) for mission in sample_missions.values())
        print(f"   Total waypoints: {total_waypoints}")
        
        for format_type, file_path in exported_files.items():
            file_size = os.path.getsize(file_path)
            print(f"   {format_type.capitalize()} format: {file_path} ({file_size} bytes)")
            
    except Exception as e:
        print(f"❌ Error exporting CSV files: {e}")
else:
    print("CSV export skipped.")
