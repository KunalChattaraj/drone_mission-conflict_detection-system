# run_ultra_optimized.py
from ultra_optimized_deconfliction import UltraOptimizedDeconflictionSystem
from mission_generator import ScalableMissionGenerator
import time

def main():
    # Initialize ultra-optimized system
    system = UltraOptimizedDeconflictionSystem()
    system.safety_buffer = 15.0
    generator = ScalableMissionGenerator(area_size=1500, altitude_range=(20, 120))
    
    print("🚀 ULTRA-OPTIMIZED High-Performance Drone Conflict Detection System")
    print("==================================================================")
    
    # Test with very large scale
    num_missions = 2000  # 2000 missions!
    print(f"Generating {num_missions} missions...")
    missions = generator.generate_missions(num_missions)
    
    print(f"\nAdding {num_missions} missions to deconfliction system...")
    start_time = time.time()
    
    # Use parallel batch addition
    results = system.batch_add_missions_parallel(missions)
    addition_time = time.time() - start_time
    
    success_count = sum(results)
    
    # Activate successful missions
    for mission, success in zip(missions, results):
        if success:
            mission.status = "active"
    
    print(f"\n✅ Successfully added {success_count}/{num_missions} missions")
    print(f"⏱️  Addition time: {addition_time:.2f} seconds")
    print(f"📊 Throughput: {num_missions/addition_time:.1f} missions/second")
    print(f"🚨 Conflicts prevented: {num_missions - success_count}")
    
    if success_count == 0:
        print("❌ No missions could be added due to conflicts. Try increasing area_size.")
        return
    
    # Start simulation
    print(f"\n🎯 Starting real-time simulation with {success_count} active missions...")
    system.simulation_running = True
    
    try:
        sim_start = time.time()
        frame_count = 0
        max_frames = 500
        
        total_conflicts_detected = 0
        
        while system.simulation_running and frame_count < max_frames:
            frame_start = time.time()
            
            # Update simulation
            system.current_sim_time += system.time_step
            system.update_drone_positions_fast()
            
            # Check for conflicts
            conflicts = system.real_time_conflict_monitoring()
            total_conflicts_detected += len(conflicts)
            
            frame_time = time.time() - frame_start
            frame_count += 1
            
            # Print status every 50 frames
            if frame_count % 50 == 0:
                active_missions = sum(1 for m in system.drone_missions.values() 
                                    if m.status == "active")
                completed = sum(1 for m in system.drone_missions.values() 
                              if m.status == "completed")
                print(f"Frame {frame_count}: {active_missions} active, {completed} completed, "
                      f"{len(conflicts)} conflicts, frame: {frame_time*1000:.1f}ms")
            
            # Maintain real-time pacing
            sleep_time = system.time_step - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Stop if all missions completed
            active_count = sum(1 for m in system.drone_missions.values() 
                             if m.status == "active")
            if active_count == 0:
                print("🏁 All missions completed!")
                break
        
        total_time = time.time() - sim_start
        print(f"\n📊 Simulation Summary:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Frames processed: {frame_count}")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        print(f"   Total conflicts detected: {total_conflicts_detected}")
        print(f"   Missions completed: {sum(1 for m in system.drone_missions.values() if m.status == 'completed')}")
        
    except KeyboardInterrupt:
        system.simulation_running = False
        print("\n⏹️  Simulation stopped by user")

if __name__ == "__main__":
    main()