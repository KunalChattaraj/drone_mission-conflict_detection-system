# mission_generator.py
import numpy as np
from optimized_deconfliction import DroneMission, Waypoint
import time
from typing import List

class ScalableMissionGenerator:
    def __init__(self, area_size=1000, altitude_range=(10, 100)):
        self.area_size = area_size
        self.altitude_range = altitude_range
        
    def generate_missions(self, n: int, max_waypoints=5) -> List[DroneMission]:
        """Generate N missions with optimized distribution"""
        print(f"Generating {n} missions...")
        start_time = time.time()
        
        missions = []
        grid_size = int(np.ceil(np.sqrt(n)))
        cell_size = self.area_size / grid_size
        
        for i in range(n):
            mission_id = f"drone_{i:06d}"
            
            # Distribute missions in grid to minimize initial conflicts
            grid_x = i % grid_size
            grid_y = i // grid_size
            
            base_x = grid_x * cell_size + np.random.uniform(0.1, 0.9) * cell_size
            base_y = grid_y * cell_size + np.random.uniform(0.1, 0.9) * cell_size
            
            waypoints = self._generate_waypoints(base_x, base_y, max_waypoints)
            
            mission = DroneMission(
                mission_id=mission_id,
                waypoints=waypoints,
                start_time=np.random.uniform(0, 60),
                end_time=np.random.uniform(300, 600),
                current_position=waypoints[0],
                speed=np.random.uniform(1.0, 3.0),
                color=self._random_color()
            )
            
            missions.append(mission)
        
        generation_time = time.time() - start_time
        print(f"Generated {n} missions in {generation_time:.2f} seconds")
        
        return missions
    
    def _generate_waypoints(self, base_x: float, base_y: float, max_waypoints: int) -> List[Waypoint]:
        """Generate waypoints with local movement"""
        num_waypoints = np.random.randint(2, max_waypoints + 1)
        waypoints = []
        
        # Start position
        start_wp = Waypoint(base_x, base_y, np.random.uniform(*self.altitude_range))
        waypoints.append(start_wp)
        
        current_x, current_y = base_x, base_y
        
        for _ in range(num_waypoints - 1):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(50, 200)
            
            new_x = current_x + distance * np.cos(angle)
            new_y = current_y + distance * np.sin(angle)
            
            # Keep within bounds
            new_x = max(0, min(self.area_size, new_x))
            new_y = max(0, min(self.area_size, new_y))
            
            waypoint = Waypoint(
                new_x, new_y, 
                np.random.uniform(*self.altitude_range)
            )
            waypoints.append(waypoint)
            
            current_x, current_y = new_x, new_y
        
        return waypoints
    
    def _random_color(self):
        """Generate random color"""
        return f"#{np.random.randint(0, 0xFFFFFF):06x}"