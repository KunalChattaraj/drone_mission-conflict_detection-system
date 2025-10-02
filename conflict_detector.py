# conflict_detector.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import math

class ConflictDetector:
    def __init__(self, safety_buffer_m: float = 10.0):
        self.safety_buffer_m = safety_buffer_m
    
    def temporal_overlap(self, start1: datetime, end1: datetime, 
                        start2: datetime, end2: datetime) -> Tuple[bool, Optional[datetime], Optional[datetime]]:
        """
        Check if two time intervals overlap.
        Returns (overlap_exists, overlap_start, overlap_end)
        """
        if pd.isna(start1) or pd.isna(end1) or pd.isna(start2) or pd.isna(end2):
            return False, None, None

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start <= overlap_end:
            return True, overlap_start, overlap_end

        return False, None, None

    def segment_to_segment_distance(self, p0: np.ndarray, p1: np.ndarray, 
                                  q0: np.ndarray, q1: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute closest points and distance between segments [p0,p1] and [q0,q1] in 3D.
        Returns (distance, s, t) where s,t are parameters in [0,1]
        """
        u = p1 - p0
        v = q1 - q0
        w0 = p0 - q0

        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)

        denom = a * c - b * b
        SMALL_NUM = 1e-12

        if denom < SMALL_NUM:
            if c <= SMALL_NUM:
                t = 0.0
            else:
                t = float(np.clip(e / c, 0.0, 1.0))
            s = 0.0
        else:
            s = (b * e - c * d) / denom
            t = (a * e - b * d) / denom
            s = float(np.clip(s, 0.0, 1.0))
            t = float(np.clip(t, 0.0, 1.0))

        cp_p = p0 + s * u
        cp_q = q0 + t * v
        distance = float(np.linalg.norm(cp_p - cp_q))
        
        return distance, s, t

    def polyline_min_distance(self, pts1: np.ndarray, pts2: np.ndarray) -> Dict[str, Any]:
        """
        Find minimum distance between two polylines with detailed information.
        Returns dictionary with distance, closest points, segments, and parameters.
        """
        n1, n2 = pts1.shape[0], pts2.shape[0]
        
        if n1 == 0 or n2 == 0:
            return {"distance": float("inf"), "conflict": False}
        
        # Handle single points
        if n1 == 1 and n2 == 1:
            distance = float(np.linalg.norm(pts1[0] - pts2[0]))
            return {
                "distance": distance,
                "conflict": distance <= self.safety_buffer_m,
                "point1": pts1[0].tolist(),
                "point2": pts2[0].tolist(),
                "segment1": 0,
                "segment2": 0,
                "param1": 0.0,
                "param2": 0.0
            }

        min_distance = float("inf")
        best_result = {}

        for i in range(max(1, n1 - 1)):
            p0 = pts1[i]
            p1 = pts1[i + 1] if i + 1 < n1 else pts1[i]
            
            for j in range(max(1, n2 - 1)):
                q0 = pts2[j]
                q1 = pts2[j + 1] if j + 1 < n2 else pts2[j]
                
                distance, s, t = self.segment_to_segment_distance(p0, p1, q0, q1)
                
                if distance < min_distance:
                    min_distance = distance
                    cp_p = p0 + s * (p1 - p0)
                    cp_q = q0 + t * (q1 - q0)
                    
                    best_result = {
                        "distance": distance,
                        "conflict": distance <= self.safety_buffer_m,
                        "point1": cp_p.tolist(),
                        "point2": cp_q.tolist(),
                        "segment1": i,
                        "segment2": j,
                        "param1": s,
                        "param2": t
                    }
                    
                    if min_distance <= 0.0:
                        best_result["distance"] = 0.0
                        best_result["conflict"] = True
                        return best_result

        return best_result if best_result else {"distance": float("inf"), "conflict": False}

    def infer_times_along_path(self, waypoints: List[List[float]], 
                             start_time: datetime, end_time: datetime) -> List[datetime]:
        """
        Infer timestamps along polyline proportional to distance.
        """
        if not waypoints:
            return []
            
        waypoints_array = np.array(waypoints)
        n = waypoints_array.shape[0]
        
        if n == 1:
            return [start_time]
        
        # Calculate cumulative distance
        deltas = np.sqrt(np.sum(np.diff(waypoints_array, axis=0) ** 2, axis=1))
        cumdist = np.concatenate(([0.0], np.cumsum(deltas)))
        total_distance = cumdist[-1]
        
        if total_distance <= 0 or math.isclose(total_distance, 0.0):
            fractions = np.linspace(0.0, 1.0, n)
        else:
            fractions = cumdist / total_distance
        
        # Convert to timestamps
        total_seconds = (end_time - start_time).total_seconds()
        times = [start_time + pd.Timedelta(seconds=float(frac * total_seconds)) 
                for frac in fractions]
        
        return times

    def detect_conflict(self, mission1: Dict[str, Any], mission2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main conflict detection between two missions.
        Returns detailed conflict information.
        """
        # Extract mission data
        waypoints1 = mission1["waypoints"]
        waypoints2 = mission2["waypoints"]
        start1, end1 = mission1["t_start"], mission1["t_end"]
        start2, end2 = mission2["t_start"], mission2["t_end"]
        
        # Convert to numpy arrays
        pts1 = np.array(waypoints1)
        pts2 = np.array(waypoints2)
        
        # Check temporal overlap first
        temporal_overlap, overlap_start, overlap_end = self.temporal_overlap(start1, end1, start2, end2)
        
        if not temporal_overlap:
            return {
                "conflict": False,
                "reason": "no_temporal_overlap",
                "min_distance": float("inf"),
                "temporal_overlap": False,
                "missions_involved": [mission1["mission_id"], mission2["mission_id"]]
            }
        
        # Check spatial distance
        distance_info = self.polyline_min_distance(pts1, pts2)
        
        if not distance_info["conflict"]:
            return {
                "conflict": False,
                "reason": "safe_distance",
                "min_distance": distance_info["distance"],
                "temporal_overlap": True,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "missions_involved": [mission1["mission_id"], mission2["mission_id"]]
            }
        
        # Detailed conflict information
        conflict_result = {
            "conflict": True,
            "reason": "spatial_violation",
            "min_distance": distance_info["distance"],
            "temporal_overlap": True,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "missions_involved": [mission1["mission_id"], mission2["mission_id"]],
            "closest_points": {
                mission1["mission_id"]: distance_info["point1"],
                mission2["mission_id"]: distance_info["point2"]
            },
            "segments": {
                mission1["mission_id"]: distance_info["segment1"],
                mission2["mission_id"]: distance_info["segment2"]
            },
            "parameters": {
                mission1["mission_id"]: distance_info["param1"],
                mission2["mission_id"]: distance_info["param2"]
            },
            "closest_times": {}  # Initialize empty, will fill if possible
        }
        
        # Add severity classification
        if distance_info["distance"] <= 0.5 * self.safety_buffer_m:
            conflict_result["severity"] = "critical"
        elif distance_info["distance"] <= 0.8 * self.safety_buffer_m:
            conflict_result["severity"] = "high"
        else:
            conflict_result["severity"] = "moderate"
        
        # Infer timestamps at closest points
        try:
            times1 = self.infer_times_along_path(waypoints1, start1, end1)
            times2 = self.infer_times_along_path(waypoints2, start2, end2)
            
            if times1 and times2:
                seg1, param1 = distance_info["segment1"], distance_info["param1"]
                seg2, param2 = distance_info["segment2"], distance_info["param2"]
                
                # Calculate time at closest point for mission1
                if seg1 < len(times1) - 1:
                    t1_start, t1_end = times1[seg1], times1[seg1 + 1]
                    time_at_closest1 = t1_start + param1 * (t1_end - t1_start)
                    conflict_result["closest_times"][mission1["mission_id"]] = time_at_closest1
                
                # Calculate time at closest point for mission2
                if seg2 < len(times2) - 1:
                    t2_start, t2_end = times2[seg2], times2[seg2 + 1]
                    time_at_closest2 = t2_start + param2 * (t2_end - t2_start)
                    conflict_result["closest_times"][mission2["mission_id"]] = time_at_closest2
                    
        except Exception as e:
            print(f"Warning: Could not infer timestamps: {e}")
        
        return conflict_result

    def check_mission_against_database(self, new_mission: Dict[str, Any], 
                                     database_missions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check a new mission against all missions in database.
        Returns list of conflicts found.
        """
        conflicts = []
        
        for mission_id, existing_mission in database_missions.items():
            if mission_id == new_mission["mission_id"]:
                continue  # Skip self-comparison
                
            conflict_info = self.detect_conflict(new_mission, existing_mission)
            
            if conflict_info["conflict"]:
                conflicts.append(conflict_info)
        
        return conflicts

    def batch_conflict_check(self, missions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check all missions against each other (for database validation).
        """
        mission_ids = list(missions.keys())
        all_conflicts = []
        
        for i in range(len(mission_ids)):
            for j in range(i + 1, len(mission_ids)):
                mission1 = missions[mission_ids[i]]
                mission2 = missions[mission_ids[j]]
                
                conflict_info = self.detect_conflict(mission1, mission2)
                
                if conflict_info["conflict"]:
                    all_conflicts.append(conflict_info)
        
        return {
            "total_checks": len(mission_ids) * (len(mission_ids) - 1) // 2,
            "conflicts_found": len(all_conflicts),
            "conflicts": all_conflicts,
            "conflict_free": len(all_conflicts) == 0
        }


# Utility functions for mission data handling
def mission_to_dataframe(mission: Dict[str, Any]) -> pd.DataFrame:
    """Convert mission dictionary to DataFrame format"""
    waypoints = mission["waypoints"]
    df = pd.DataFrame(waypoints, columns=["x", "y", "z"])
    df["start_time"] = pd.to_datetime(mission["t_start"]).tz_localize("UTC")
    df["end_time"] = pd.to_datetime(mission["t_end"]).tz_localize("UTC")
    return df

def create_sample_mission(mission_id: str, waypoints: List[List[float]], 
                         start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Create a mission dictionary"""
    return {
        "mission_id": mission_id,
        "waypoints": waypoints,
        "t_start": start_time,
        "t_end": end_time
    }


# Example usage and testing
if __name__ == "__main__":
    # Create detector
    detector = ConflictDetector(safety_buffer_m=10.0)
    
    # Create sample missions
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    mission1 = create_sample_mission(
        "mission_001",
        [[0, 0, 100], [100, 50, 110], [200, 100, 120]],
        base_time,
        base_time + pd.Timedelta(minutes=30)
    )
    
    mission2 = create_sample_mission(
        "mission_002", 
        [[5, 5, 102], [105, 55, 108], [205, 105, 118]],
        base_time + pd.Timedelta(minutes=5),
        base_time + pd.Timedelta(minutes=25)
    )
    
    # Detect conflict
    conflict = detector.detect_conflict(mission1, mission2)
    
    print("Conflict Detection Result:")
    print("=" * 50)
    for key, value in conflict.items():
        print(f"{key}: {value}")