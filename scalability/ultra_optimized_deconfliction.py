# ultra_optimized_deconfliction.py
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
from collections import defaultdict

@dataclass
class Waypoint:
    x: float
    y: float
    z: float = 0.0
    timestamp: Optional[float] = None

@dataclass 
class DroneMission:
    mission_id: str
    waypoints: List[Waypoint]
    start_time: float
    end_time: float
    current_position: Waypoint
    current_waypoint_index: int = 0
    speed: float = 1.0
    status: str = "pending"
    color: str = "blue"
    trajectory_cache: Optional[np.ndarray] = None

class UltraOptimizedDeconflictionSystem:
    def __init__(self):
        self.drone_missions: Dict[str, DroneMission] = {}
        self.safety_buffer: float = 15.0
        self.time_step: float = 0.1
        self.current_sim_time: float = 0.0
        self.conflicts = []
        self.simulation_running: bool = False
        
        # Performance optimization
        self.trajectory_cache: Dict[str, np.ndarray] = {}
        self.mission_ids: List[str] = []
        self.spatial_index: Dict[Tuple[int, int], List[str]] = {}
        self.grid_size: int = 50
        
        # Real-time optimization
        self.last_conflict_check_time: float = 0.0
        self.conflict_check_interval: float = 1.0  # Check conflicts every 1 second
        
    def _get_grid_key(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates to grid key"""
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        return (grid_x, grid_y)
    
    def _get_grid_cells_for_trajectory(self, trajectory: np.ndarray) -> set:
        """Get all grid cells that a trajectory passes through"""
        cells = set()
        for point in trajectory:
            cells.add(self._get_grid_key(point[0], point[1]))
        return cells
    
    def add_mission(self, mission: DroneMission) -> bool:
        """Ultra-optimized mission addition with spatial indexing"""
        trajectory = self._ultra_fast_trajectory(mission)
        mission.trajectory_cache = trajectory
        self.trajectory_cache[mission.mission_id] = trajectory
        
        if not self._ultra_fast_conflict_check_with_index(mission, trajectory):
            self.drone_missions[mission.mission_id] = mission
            self.mission_ids.append(mission.mission_id)
            
            grid_cells = self._get_grid_cells_for_trajectory(trajectory)
            for cell in grid_cells:
                if cell not in self.spatial_index:
                    self.spatial_index[cell] = []
                self.spatial_index[cell].append(mission.mission_id)
            
            return True
        return False
    
    def _ultra_fast_trajectory(self, mission: DroneMission) -> np.ndarray:
        """Ultra-fast trajectory with minimal points"""
        waypoints = np.array([[wp.x, wp.y, wp.z] for wp in mission.waypoints])
        num_segments = len(waypoints) - 1
        
        if num_segments == 0:
            return np.array([[waypoints[0][0], waypoints[0][1], waypoints[0][2], mission.start_time]])
        
        total_points = min(30, max(10, num_segments * 5))
        trajectory = []
        current_time = mission.start_time
        
        for i in range(num_segments):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            distance = np.linalg.norm(wp2 - wp1)
            time_needed = distance / mission.speed
            
            points_per_segment = max(3, total_points // num_segments)
            t_values = np.linspace(0, 1, points_per_segment)
            points = wp1 + t_values[:, np.newaxis] * (wp2 - wp1)
            times = current_time + t_values * time_needed
            
            for j in range(points_per_segment):
                trajectory.append([points[j][0], points[j][1], points[j][2], times[j]])
            
            current_time += time_needed
        
        return np.array(trajectory)
    
    def _ultra_fast_conflict_check_with_index(self, new_mission: DroneMission, new_trajectory: np.ndarray) -> bool:
        """Ultra-fast conflict checking using spatial index"""
        if not self.drone_missions:
            return False
        
        new_cells = self._get_grid_cells_for_trajectory(new_trajectory)
        
        missions_to_check = set()
        for cell in new_cells:
            if cell in self.spatial_index:
                missions_to_check.update(self.spatial_index[cell])
        
        for cell in list(new_cells):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_cell = (cell[0] + dx, cell[1] + dy)
                    if neighbor_cell in self.spatial_index:
                        missions_to_check.update(self.spatial_index[neighbor_cell])
        
        for mission_id in missions_to_check:
            existing_trajectory = self.trajectory_cache[mission_id]
            if self._quick_trajectory_conflict(new_trajectory, existing_trajectory):
                return True
        
        return False
    
    def _quick_trajectory_conflict(self, traj1: np.ndarray, traj2: np.ndarray) -> bool:
        """Very quick trajectory conflict check"""
        key_indices1 = self._get_key_indices(traj1)
        key_indices2 = self._get_key_indices(traj2)
        
        for idx1 in key_indices1:
            point1 = traj1[idx1]
            time1 = point1[3]
            
            for idx2 in key_indices2:
                point2 = traj2[idx2]
                time2 = point2[3]
                
                if abs(time1 - time2) < self.time_step * 3:
                    distance = np.linalg.norm(point1[:3] - point2[:3])
                    if distance <= self.safety_buffer:
                        return True
        
        return False
    
    def _get_key_indices(self, trajectory: np.ndarray) -> List[int]:
        """Get key indices (start, middle, end)"""
        n = len(trajectory)
        if n <= 3:
            return list(range(n))
        return [0, n//2, n-1]

    def real_time_conflict_monitoring(self):
        """OPTIMIZED real-time conflict monitoring with spatial indexing"""
        active_missions = [
            (mission_id, mission) for mission_id, mission in self.drone_missions.items() 
            if mission.status == "active"
        ]
        
        if len(active_missions) < 2:
            return []
        
        # Build spatial index for current positions
        current_spatial_index = defaultdict(list)
        mission_data = {}
        
        for mission_id, mission in active_missions:
            pos = mission.current_position
            grid_key = self._get_grid_key(pos.x, pos.y)
            current_spatial_index[grid_key].append(mission_id)
            mission_data[mission_id] = (pos.x, pos.y, pos.z)
        
        conflicts = []
        
        # Only check missions in the same or adjacent grid cells
        for grid_key, mission_ids_in_cell in current_spatial_index.items():
            # Check within same cell
            for i in range(len(mission_ids_in_cell)):
                for j in range(i + 1, len(mission_ids_in_cell)):
                    mission_id1 = mission_ids_in_cell[i]
                    mission_id2 = mission_ids_in_cell[j]
                    
                    x1, y1, z1 = mission_data[mission_id1]
                    x2, y2, z2 = mission_data[mission_id2]
                    
                    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    if distance <= self.safety_buffer:
                        conflicts.append({
                            'drone1': mission_id1,
                            'drone2': mission_id2,
                            'distance': distance,
                            'time': self.current_sim_time
                        })
            
            # Check adjacent cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip same cell (already checked)
                    
                    neighbor_key = (grid_key[0] + dx, grid_key[1] + dy)
                    if neighbor_key in current_spatial_index:
                        for mission_id1 in mission_ids_in_cell:
                            for mission_id2 in current_spatial_index[neighbor_key]:
                                x1, y1, z1 = mission_data[mission_id1]
                                x2, y2, z2 = mission_data[mission_id2]
                                
                                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                                if distance <= self.safety_buffer:
                                    conflicts.append({
                                        'drone1': mission_id1,
                                        'drone2': mission_id2,
                                        'distance': distance,
                                        'time': self.current_sim_time
                                    })
        
        return conflicts
    
    def update_drone_positions_fast(self):
        """Optimized position updates"""
        for mission in self.drone_missions.values():
            if mission.status != "active":
                continue
            self._update_single_drone_position(mission)
    
    def _update_single_drone_position(self, mission: DroneMission):
        """Fast single drone position update"""
        if mission.current_waypoint_index >= len(mission.waypoints) - 1:
            mission.status = "completed"
            return
        
        current_idx = mission.current_waypoint_index
        current_wp = mission.waypoints[current_idx]
        next_wp = mission.waypoints[current_idx + 1]
        
        dx = next_wp.x - current_wp.x
        dy = next_wp.y - current_wp.y
        dz = next_wp.z - current_wp.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance > 0:
            move_distance = mission.speed * self.time_step
            if move_distance >= distance:
                mission.current_waypoint_index += 1
                mission.current_position = next_wp
            else:
                ratio = move_distance / distance
                mission.current_position.x = current_wp.x + dx * ratio
                mission.current_position.y = current_wp.y + dy * ratio
                mission.current_position.z = current_wp.z + dz * ratio
    
    def batch_add_missions_parallel(self, missions: List[DroneMission]) -> List[bool]:
        """Parallel batch mission addition"""
        batch_size = 50
        results = []
        
        for i in range(0, len(missions), batch_size):
            batch = missions[i:i + batch_size]
            batch_results = [self.add_mission(mission) for mission in batch]
            results.extend(batch_results)
            
            if (i + batch_size) % 200 == 0:
                print(f"  Processed {min(i + batch_size, len(missions))}/{len(missions)} missions...")
        
        return results