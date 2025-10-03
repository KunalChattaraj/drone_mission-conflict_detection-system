# Drone Mission Conflict Detection System  

## 🚀 Overview  
A comprehensive drone mission planning and conflict detection system designed to ensure **safe airspace operations**. The system checks for **temporal and spatial conflicts** between multiple drone missions, using deterministic algorithms with optional AI enhancements for predictive analysis.  

## 📋 Features  

- **Mission Generation**: Create realistic drone mission datasets  
- **Mission Management**: Store and manage missions in multiple formats (compact/expanded CSV)  
- **Temporal Conflict Detection**: Efficient interval checks (O(1)) for overlapping mission windows  
- **Spatial Conflict Detection**: 3D line-segment distance checks with configurable safety buffers  
- **Temporal Interpolation**: Continuous trajectory evaluation between waypoints for accurate conflict detection  
- **Visualization**: Interactive 3D plots and animations of flight paths and conflict points  
- **Multiple Input Formats**: Manual input or CSV-based batch processing  
- **Scalability Ready**: Architecture designed to migrate to spatial-temporal databases, distributed processing, and real-time streaming  
- **Testing Suite**: Comprehensive coverage of unit, integration, and edge cases to ensure safety-critical correctness  

## 🛠 Installation  

### Prerequisites  
- Python 3.8+  
- pip (Python package manager)  

### Required Packages  

```bash
pip install pandas numpy matplotlib
```

Step 1: Generate Mission Files
```bash
python simulated_mission.py
```
Generates:
mission_compact.csv → One row per mission
missions_expanded.csv → One row per waypoint

Step 2: Create Test Missions
```bash
python create_test_missions.py
```
Generates sample scenarios in test_missions/:
- safe_mission.csv – No conflict
- critical_conflict_mission.csv – Paths within safety buffer
- moderate_conflict_mission.csv – Marginal separation
- different_time_mission.csv – Same path, different times
- high_altitude_mission.csv – Same path, different altitudes

Step 3: Run the Conflict Detection System

```bash
python io.py
```
You’ll see:
- Option 1: Manual Input → Enter waypoints interactively, specify mission timing, get immediate results
- Option 2: CSV Input → Run batch checks using predefined files

Step 4: View Results & Visualizations
- ✅ Text-based conflict reports
- 🎨 Optional 3D trajectory visualization
- 📊 Detailed logs of conflict points
- 💡 Safety resolution suggestions

<img width="3840" height="488" alt="Untitled diagram _ Mermaid Chart-2025-10-02-194222" src="https://github.com/user-attachments/assets/51fc487a-916c-4cbb-9e09-07036b5b6b51" />

🧪 Testing Strategy
- Unit Tests: Temporal overlap, spatial distance (parallel, coincident, zero-length), safety buffer boundaries
- Integration Tests: End-to-end workflow (mission input → detection → CSV/report → visualization)
- Edge Cases: Identical timings, nested windows, vertical separation, malformed coordinates
- Representative Missions: Safe, critical conflict, temporal separation, vertical separation
- Philosophy: “Test the edges, the middle takes care of itself.”

📈 Scaling Considerations
- Database Migration: From CSV → PostGIS or Redis with spatial indexing
- Temporal Interpolation: Continuous trajectory checks for accurate time-based conflicts
- Spatial Indexing: R-tree / Quadtree for O(n log n) candidate filtering
- Parallel Processing: Distributed mission batches (map-reduce style)
- Infrastructure: Kubernetes + Redis + Kafka for real-time scalable deployment
- Safety & Compliance: Audit logging, redundancy layers, certification-ready architecture

# 🛡️ Safety Buffer System

## Safety Buffer Zones

| Severity Level | Distance Range | Description | Visual Indicator |
|---------------|----------------|-------------|------------------|
| 🚨 **Critical** | ≤ 5m | Immediate danger - High collision risk | Solid red spheres |
| ⚠️ **High** | 5m - 8m | High risk - Requires immediate attention | Dark orange zones |
| 📍 **Moderate** | 8m - 10m | Moderate risk - Proceed with caution | Light orange zones |
| ✅ **Safe** | > 10m | No conflict - Clear for operation | No visual warning |

## How It Works

- **10m Safety Buffer**: Default protected airspace around each drone
- **Real-time Monitoring**: Continuous 3D distance calculations
- **Progressive Alerts**: Escalating warnings based on separation distance
- **Visual Feedback**: Color-coded spheres in 3D visualizations

## Configuration

```python
# Adjust safety buffer in your code
checker = MissionConflictChecker(safety_buffer_m=15.0)  # 15m buffer
