# optimized_deconfliction.py
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

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

class HighPerformanceDeconflictionSystem:
    def __init__(self):
        self.drone_missions: Dict[str, DroneMission] = {}
        self.safety_buffer: float = 10.0
        self.time_step: float = 0.1
        self.current_sim_time: float = 0.0
        self.conflicts = []
        self.simulation_running: bool = False
        
        # Performance optimization
        self.trajectory_cache: Dict[str, np.ndarray] = {}
        self.mission_ids: List[str] = []
        
        # Parallel processing
        self.num_workers = min(mp.cpu_count(), 8)
        
    def add_mission(self, mission: DroneMission) -> bool:
        """Optimized mission addition with faster conflict checking"""
        # Pre-compute trajectory
        trajectory = self._precompute_trajectory_fast(mission)
        mission.trajectory_cache = trajectory
        self.trajectory_cache[mission.mission_id] = trajectory
        
        # Fast conflict check
        if not self._ultra_fast_conflict_check(mission, trajectory):
            self.drone_missions[mission.mission_id] = mission
            self.mission_ids.append(mission.mission_id)
            return True
        return False
    
    def _precompute_trajectory_fast(self, mission: DroneMission) -> np.ndarray:
        """Optimized trajectory precomputation with fewer points"""
        waypoints = np.array([[wp.x, wp.y, wp.z] for wp in mission.waypoints])
        num_segments = len(waypoints) - 1
        
        if num_segments == 0:
            return np.array([[waypoints[0][0], waypoints[0][1], waypoints[0][2], mission.start_time]])
        
        # Use fewer points for faster computation
        total_points = min(100, int((mission.end_time - mission.start_time) / self.time_step))
        points_per_segment = max(10, total_points // num_segments)
        
        trajectory = []
        current_time = mission.start_time
        
        for i in range(num_segments):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            distance = np.linalg.norm(wp2 - wp1)
            time_needed = distance / mission.speed
            
            # Use fewer interpolation points
            t_values = np.linspace(0, 1, points_per_segment)
            points = wp1 + t_values[:, np.newaxis] * (wp2 - wp1)
            times = current_time + t_values * time_needed
            
            for j in range(points_per_segment):
                trajectory.append([points[j][0], points[j][1], points[j][2], times[j]])
            
            current_time += time_needed
        
        return np.array(trajectory)
    
    def _ultra_fast_conflict_check(self, new_mission: DroneMission, new_trajectory: np.ndarray) -> bool:
        """Ultra-fast conflict checking using spatial and temporal pruning"""
        if not self.drone_missions:
            return False
        
        new_time_range = (new_trajectory[0, 3], new_trajectory[-1, 3])
        new_spatial_range = self._get_spatial_range(new_trajectory)
        
        # Check against existing missions
        for mission_id in self.mission_ids:
            existing_mission = self.drone_missions[mission_id]
            if existing_mission.status in ["completed", "paused"]:
                continue
                
            existing_trajectory = self.trajectory_cache[mission_id]
            
            # Quick spatial and temporal pruning
            if not self._quick_prune_check(new_trajectory, existing_trajectory, new_time_range, new_spatial_range):
                # Only do detailed check if pruning doesn't eliminate conflict possibility
                if self._fast_trajectory_conflict(new_trajectory, existing_trajectory):
                    return True
        
        return False
    
    def _get_spatial_range(self, trajectory: np.ndarray) -> Tuple[float, float, float, float]:
        """Get spatial bounding box for quick pruning"""
        x_min, x_max = np.min(trajectory[:, 0]), np.max(trajectory[:, 0])
        y_min, y_max = np.min(trajectory[:, 1]), np.max(trajectory[:, 1])
        return (x_min, x_max, y_min, y_max)
    
    def _quick_prune_check(self, traj1: np.ndarray, traj2: np.ndarray, 
                          time_range1: Tuple[float, float], spatial_range1: Tuple[float, float, float, float]) -> bool:
        """Quick pruning based on spatial and temporal bounds"""
        # Temporal pruning
        time_range2 = (traj2[0, 3], traj2[-1, 3])
        if time_range1[1] < time_range2[0] or time_range1[0] > time_range2[1]:
            return False  # No time overlap
        
        # Spatial pruning with expanded bounds for safety buffer
        buffer_expansion = self.safety_buffer * 2
        x_min2, x_max2 = np.min(traj2[:, 0]), np.max(traj2[:, 0])
        y_min2, y_max2 = np.min(traj2[:, 1]), np.max(traj2[:, 1])
        
        # Check if bounding boxes overlap (with safety buffer)
        if (spatial_range1[1] + buffer_expansion < x_min2 or 
            spatial_range1[0] - buffer_expansion > x_max2 or
            spatial_range1[3] + buffer_expansion < y_min2 or 
            spatial_range1[2] - buffer_expansion > y_max2):
            return False  # No spatial overlap
        
        return True  # Potential conflict, need detailed check
    
    def _fast_trajectory_conflict(self, traj1: np.ndarray, traj2: np.ndarray) -> bool:
        """Fast conflict detection using sampled time points"""
        # Sample key points from both trajectories
        sample_indices1 = self._get_sample_indices(traj1)
        sample_indices2 = self._get_sample_indices(traj2)
        
        # Check sampled points
        for idx1 in sample_indices1:
            point1 = traj1[idx1]
            time1 = point1[3]
            
            # Find closest point in time from trajectory 2
            closest_idx2 = self._find_closest_time_index(traj2, time1)
            if closest_idx2 is not None:
                point2 = traj2[closest_idx2]
                time_diff = abs(point2[3] - time1)
                
                if time_diff <= self.time_step * 2:  # Slightly larger time window
                    distance = np.linalg.norm(point1[:3] - point2[:3])
                    if distance <= self.safety_buffer:
                        return True
        
        return False
    
    def _get_sample_indices(self, trajectory: np.ndarray, max_samples: int = 20) -> List[int]:
        """Get evenly spaced sample indices from trajectory"""
        n = len(trajectory)
        if n <= max_samples:
            return list(range(n))
        
        step = max(1, n // max_samples)
        return list(range(0, n, step))
    
    def _find_closest_time_index(self, trajectory: np.ndarray, target_time: float) -> Optional[int]:
        """Find index of point closest to target time"""
        times = trajectory[:, 3]
        idx = np.searchsorted(times, target_time)
        
        if idx == 0:
            return 0
        elif idx >= len(times):
            return len(times) - 1
        else:
            # Check both surrounding points
            time_diff_prev = abs(times[idx - 1] - target_time)
            time_diff_next = abs(times[idx] - target_time)
            return idx - 1 if time_diff_prev < time_diff_next else idx

    # Keep the rest of the methods the same (they're already fast)
    def real_time_conflict_monitoring(self):
        """Optimized real-time conflict monitoring"""
        active_missions = [
            (mission_id, mission) for mission_id, mission in self.drone_missions.items() 
            if mission.status == "active"
        ]
        
        if len(active_missions) < 2:
            return []
        
        positions = []
        mission_ids = []
        
        for mission_id, mission in active_missions:
            positions.append([mission.current_position.x, mission.current_position.y, mission.current_position.z])
            mission_ids.append(mission_id)
        
        positions = np.array(positions)
        return self._fast_proximity_check(positions, mission_ids)
    
    def _fast_proximity_check(self, positions: np.ndarray, mission_ids: List[str]) -> List[dict]:
        """Fast proximity checking"""
        n = len(positions)
        conflicts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= self.safety_buffer:
                    conflicts.append({
                        'drone1': mission_ids[i],
                        'drone2': mission_ids[j],
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
    
    def batch_add_missions(self, missions: List[DroneMission]) -> List[bool]:
        """Add multiple missions in batch"""
        return [self.add_mission(mission) for mission in missions]