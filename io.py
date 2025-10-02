# io.py
import pandas as pd
from datetime import datetime
import os
from conflict_detector import ConflictDetector
from mission_visualizer import MissionVisualizer, quick_visualize

class MissionConflictChecker:
    def __init__(self, safety_buffer_m=10.0):
        self.safety_buffer_m = safety_buffer_m
        self.detector = ConflictDetector(safety_buffer_m=safety_buffer_m)
        self.visualizer = MissionVisualizer(safety_buffer_m=safety_buffer_m)
        self.simulated_missions = self.load_simulated_missions()
    
    def load_simulated_missions(self):
        """Load existing missions from missions_compact.csv file"""
        if not os.path.exists("missions_compact.csv"):
            print("❌ missions_compact.csv not found!")
            return {}
        
        try:
            df = pd.read_csv("missions_compact.csv")
            missions = {}
            for _, row in df.iterrows():
                mission_id = row['mission_id']
                waypoints = eval(row['waypoints'])  # Convert string to list of lists
                missions[mission_id] = {
                    'mission_id': mission_id,
                    'waypoints': waypoints,
                    't_start': pd.to_datetime(row['start_time']),
                    't_end': pd.to_datetime(row['end_time'])
                }
            print(f"✅ Loaded {len(missions)} simulated missions from missions_compact.csv")
            return missions
        except Exception as e:
            print(f"❌ Error loading missions_compact.csv: {e}")
            return {}
    
    def get_user_mission(self):
        """Get mission details from user input"""
        print("\n" + "="*50)
        print("🚀 NEW MISSION INPUT")
        print("="*50)
        
        print("Mission ID will be assigned automatically")
        
        # Get waypoints
        print("\n📍 Enter Waypoints (x, y, z coordinates):")
        print("   Format: x, y, z (separated by commas)")
        print("   Type 'done' when finished")
        
        waypoints = []
        while True:
            coord_input = input(f"   Waypoint {len(waypoints) + 1}: ").strip()
            if coord_input.lower() == 'done':
                if len(waypoints) < 2:
                    print("   ❌ Need at least 2 waypoints. Please enter more.")
                    continue
                break
            
            try:
                coords = [float(x.strip()) for x in coord_input.split(',')]
                if len(coords) != 3:
                    print("   ❌ Please enter exactly 3 coordinates (x, y, z)")
                    continue
                waypoints.append(coords)
                print(f"   ✅ Added: {coords}")
            except ValueError:
                print("   ❌ Invalid coordinates. Use format: x, y, z")
        
        # Get start time
        while True:
            start_input = input("\n⏰ Enter start time (YYYY-MM-DD HH:MM:SS): ").strip()
            try:
                t_start = pd.to_datetime(start_input)
                break
            except:
                print("   ❌ Invalid datetime format. Use: YYYY-MM-DD HH:MM:SS")
        
        # Get end time
        while True:
            end_input = input("⏰ Enter end time (YYYY-MM-DD HH:MM:SS): ").strip()
            try:
                t_end = pd.to_datetime(end_input)
                if t_end <= t_start:
                    print("   ❌ End time must be after start time")
                    continue
                break
            except:
                print("   ❌ Invalid datetime format. Use: YYYY-MM-DD HH:MM:SS")
        
        return {
            'waypoints': waypoints,
            't_start': t_start,
            't_end': t_end
        }
    
    def input_from_csv(self, csv_file):
        """Read mission data from CSV file (expanded format)"""
        try:
            df = pd.read_csv(csv_file)
            
            # Check if it's compact format (has waypoints column) or expanded format
            if 'waypoints' in df.columns:
                # Compact format - one row per mission
                if len(df) > 1:
                    print("⚠️  Multiple missions in CSV. Using first mission.")
                row = df.iloc[0]
                waypoints = eval(row['waypoints'])
                t_start = pd.to_datetime(row['start_time'])
                t_end = pd.to_datetime(row['end_time'])
            else:
                # Expanded format - multiple rows per mission
                # Check required columns
                required_cols = ['x', 'y', 'z', 'start_time', 'end_time']
                if not all(col in df.columns for col in required_cols):
                    print(f"❌ CSV must contain columns: {required_cols}")
                    return None
                
                waypoints = []
                for _, row in df.iterrows():
                    waypoints.append([row['x'], row['y'], row['z']])
                
                # Use first row for start/end times
                t_start = pd.to_datetime(df['start_time'].iloc[0])
                t_end = pd.to_datetime(df['end_time'].iloc[0])
            
            return {
                'mission_id': 'user_mission',
                'waypoints': waypoints,
                't_start': t_start,
                't_end': t_end
            }
            
        except Exception as e:
            print(f"❌ Error reading CSV file: {e}")
            return None
    
    def check_conflicts(self, user_mission):
        """Check user mission against all simulated missions"""
        print(f"\n🔍 Checking conflicts against {len(self.simulated_missions)} simulated missions...")
        
        conflicts = []
        for sim_id, sim_mission in self.simulated_missions.items():
            conflict_info = self.detector.detect_conflict(user_mission, sim_mission)
            
            if conflict_info['conflict']:
                conflicts.append({
                    'simulated_mission': sim_id,
                    'min_distance': conflict_info['min_distance'],
                    'severity': conflict_info.get('severity', 'unknown'),
                    'temporal_overlap': conflict_info['temporal_overlap'],
                    'closest_points': conflict_info.get('closest_points', {}),
                    'reason': conflict_info['reason']
                })
                print(f"   🚨 Conflict with {sim_id}: {conflict_info['min_distance']:.2f}m ({conflict_info.get('severity', 'unknown')})")
            else:
                print(f"   ✅ No conflict with {sim_id}")
        
        return conflicts
    
    def print_conflict_summary(self, user_mission, conflicts):
        """Print comprehensive conflict summary"""
        print("\n" + "="*60)
        print("📊 CONFLICT ANALYSIS SUMMARY")
        print("="*60)
        
        # Mission overview
        print(f"\n📍 USER MISSION OVERVIEW:")
        print(f"   Waypoints: {len(user_mission['waypoints'])}")
        print(f"   Time: {user_mission['t_start']} to {user_mission['t_end']}")
        print(f"   Duration: {(user_mission['t_end'] - user_mission['t_start']).total_seconds()/60:.1f} minutes")
        
        # Show first and last waypoint
        if user_mission['waypoints']:
            print(f"   First waypoint: {user_mission['waypoints'][0]}")
            print(f"   Last waypoint: {user_mission['waypoints'][-1]}")
        
        # Conflict statistics
        print(f"\n📈 CONFLICT STATISTICS:")
        print(f"   Simulated missions checked: {len(self.simulated_missions)}")
        print(f"   Missions in database: {list(self.simulated_missions.keys())}")
        print(f"   Conflicts detected: {len(conflicts)}")
        
        if conflicts:
            critical_conflicts = [c for c in conflicts if c['severity'] == 'critical']
            high_conflicts = [c for c in conflicts if c['severity'] == 'high']
            moderate_conflicts = [c for c in conflicts if c['severity'] == 'moderate']
            
            print(f"   🚨 Critical: {len(critical_conflicts)}")
            print(f"   ⚠️  High: {len(high_conflicts)}")
            print(f"   📍 Moderate: {len(moderate_conflicts)}")
            
            # Overall recommendation
            if critical_conflicts:
                print(f"\n❌ RECOMMENDATION: MISSION REJECTED - Critical conflicts detected")
            elif high_conflicts:
                print(f"\n⚠️  RECOMMENDATION: MISSION RISKY - High severity conflicts")
            else:
                print(f"\n📍 RECOMMENDATION: Proceed with caution - Moderate conflicts")
        else:
            print(f"\n✅ RECOMMENDATION: MISSION APPROVED - No conflicts detected")
        
        # Detailed conflict breakdown
        if conflicts:
            print(f"\n🔍 DETAILED CONFLICT BREAKDOWN:")
            print("-" * 50)
            
            for i, conflict in enumerate(conflicts, 1):
                print(f"\nConflict #{i}:")
                print(f"  Simulated Mission: {conflict['simulated_mission']}")
                print(f"  Minimum Distance: {conflict['min_distance']:.2f}m")
                print(f"  Severity: {conflict['severity'].upper()}")
                print(f"  Temporal Overlap: {conflict['temporal_overlap']}")
                print(f"  Reason: {conflict['reason']}")
                
                if 'closest_points' in conflict and conflict['closest_points']:
                    print("  Closest Points:")
                    for mission_id, point in conflict['closest_points'].items():
                        if mission_id == 'user_mission':
                            print(f"    Your Mission: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
                        else:
                            print(f"    {mission_id}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
    
    def generate_visualizations(self, user_mission, conflicts):
        """Generate various visualizations for the mission analysis"""
        print(f"\n🎨 GENERATING VISUALIZATIONS...")
        
        # Create output directory for visualizations
        os.makedirs("visualizations", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Ask user what type of visualization they want
            print("\n📊 Choose visualization type:")
            print("1. Static 3D Plot (Recommended for reports)")
            print("2. Animated 3D Plot (Shows mission progression)")
            print("3. Analysis Dashboard (Multiple views + statistics)")
            print("4. All visualizations")
            
            viz_choice = input("\nEnter choice (1-4): ").strip()
            
            if viz_choice == '1' or viz_choice == '4':
                # Static 3D Plot
                static_path = f"visualizations/conflict_3d_{timestamp}.png"
                print("📈 Generating static 3D visualization...")
                self.visualizer.create_static_3d_plot(
                    user_mission, 
                    self.simulated_missions, 
                    conflicts, 
                    save_path=static_path
                )
            
            if viz_choice == '2' or viz_choice == '4':
                # Animated 3D Plot
                if conflicts:  # Only create animation if there are conflicts (more interesting)
                    animated_path = f"visualizations/conflict_animation_{timestamp}.gif"
                    print("🎬 Generating animated visualization...")
                    self.visualizer.create_animated_3d_plot(
                        user_mission,
                        self.simulated_missions,
                        conflicts,
                        save_path=animated_path
                    )
                else:
                    print("ℹ️  Skipping animation - no conflicts to visualize")
            
            if viz_choice == '3' or viz_choice == '4':
                # Analysis Dashboard
                analysis_path = f"visualizations/conflict_analysis_{timestamp}.png"
                print("📋 Generating analysis dashboard...")
                self.visualizer.create_conflict_analysis_plot(
                    user_mission,
                    self.simulated_missions,
                    conflicts,
                    save_path=analysis_path
                )
            
            print(f"\n💾 Visualizations saved in: visualizations/")
            print("   You can find the generated PNG and GIF files there!")
            
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
            print("   But conflict analysis is still complete!")
            # Try quick visualization as fallback
            try:
                print("🔄 Attempting quick visualization...")
                quick_visualize(user_mission, self.simulated_missions, conflicts, plot_type='static')
            except Exception as e2:
                print(f"❌ Quick visualization also failed: {e2}")

def main():
    """Main function to run the conflict checking system"""
    print("="*60)
    print("🎯 DRONE MISSION CONFLICT CHECKER")
    print("="*60)
    print("Checking against missions in missions_compact.csv")
    
    # Initialize checker
    checker = MissionConflictChecker(safety_buffer_m=10.0)
    
    if not checker.simulated_missions:
        print("❌ Cannot proceed without missions_compact.csv")
        return
    
    # Input method selection
    print("\nChoose input method:")
    print("1. Manual input")
    print("2. CSV file input (expanded or compact format)")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    user_mission = None
    
    if choice == '1':
        user_mission = checker.get_user_mission()
    elif choice == '2':
        csv_file = input("Enter CSV file path: ").strip()
        user_mission = checker.input_from_csv(csv_file)
    else:
        print("❌ Invalid choice")
        return
    
    if user_mission is None:
        print("❌ Failed to get mission data")
        return
    
    # Check for conflicts
    conflicts = checker.check_conflicts(user_mission)
    
    # Print summary
    checker.print_conflict_summary(user_mission, conflicts)
    
    # Generate visualizations
    if conflicts:
        print(f"\n🎨 This mission has {len(conflicts)} conflict(s). Visualizations can help understand them.")
    else:
        print(f"\n🎨 This mission is conflict-free! Visualizations can show the safe paths.")
    
    visualize = input("\nGenerate visualizations? (y/n): ").lower().strip()
    if visualize in ['y', 'yes']:
        checker.generate_visualizations(user_mission, conflicts)
    else:
        print("📊 Visualization skipped.")
    
    # Additional suggestions if conflicts exist
    if conflicts:
        print(f"\n💡 SUGGESTIONS TO RESOLVE CONFLICTS:")
        print("   1. Adjust waypoints to increase separation distance")
        print("   2. Change mission timing to avoid temporal overlap") 
        print("   3. Increase altitude separation")
        print("   4. Reroute around conflict areas")
        print("   5. Stagger mission start times")
        
        # Show specific suggestions based on conflict severity
        critical_conflicts = [c for c in conflicts if c['severity'] == 'critical']
        if critical_conflicts:
            print(f"\n🚨 URGENT: {len(critical_conflicts)} critical conflicts require immediate attention!")
            print("   Consider complete mission replanning or significant timing changes.")
    
    print(f"\n" + "="*60)
    print("✅ MISSION ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()