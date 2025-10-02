# mission_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime, timedelta
import matplotlib.colors as mcolors
from conflict_detector import ConflictDetector

class MissionVisualizer:
    def __init__(self, safety_buffer_m=10.0):
        self.safety_buffer_m = safety_buffer_m
        self.detector = ConflictDetector(safety_buffer_m=safety_buffer_m)
        self.animation_artists = []
    
    def create_static_3d_plot(self, user_mission, simulated_missions, conflicts, save_path=None):
        """
        Create a static 3D visualization of all missions with conflict highlights
        """
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot user mission (primary)
        self._plot_mission(ax, user_mission, 'User Mission', 'blue', 'o-', linewidth=3, markersize=8)
        
        # Plot simulated missions
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (mission_id, mission) in enumerate(simulated_missions.items()):
            color = colors[i % len(colors)]
            self._plot_mission(ax, mission, mission_id, color, '^-', linewidth=2, markersize=6)
        
        # Highlight conflict areas
        self._highlight_conflicts(ax, conflicts)
        
        # Add safety buffer visualization
        self._add_safety_buffer_visualization(ax, user_mission)
        
        # Customize plot
        self._customize_plot(ax, user_mission, simulated_missions)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Static 3D plot saved: {save_path}")
        
        plt.show()
    
    def create_animated_3d_plot(self, user_mission, simulated_missions, conflicts, save_path=None):
        """
        Create an animated 3D visualization showing mission progression (constant speed)
        """
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Store missions for animation
        self.user_mission = user_mission
        self.simulated_missions = simulated_missions
        self.conflicts = conflicts
        
        # Setup animation base plot
        self._setup_animation_plot(ax, user_mission, simulated_missions)
        
        # Create animation with constant speed assumption
        anim = animation.FuncAnimation(
            fig, self._update_animation,
            fargs=(ax,),
            frames=50,  # Reduced frames for better performance
            interval=200,  # Slower interval for better viewing
            blit=False, 
            repeat=True
        )
        
        plt.tight_layout()
        
        if save_path:
            try:
                anim.save(save_path, writer='pillow', fps=5)  # Lower FPS for GIF
                print(f"💾 Animation saved: {save_path}")
            except Exception as e:
                print(f"⚠️  Could not save animation: {e}")
                print("   But you can still view it interactively!")
        
        plt.show()
        return anim
    
    def create_conflict_analysis_plot(self, user_mission, simulated_missions, conflicts, save_path=None):
        """
        Create a comprehensive analysis plot with multiple views
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 3D View
        ax1 = fig.add_subplot(221, projection='3d')
        self._create_3d_view(ax1, user_mission, simulated_missions, conflicts)
        
        # Top View (2D)
        ax2 = fig.add_subplot(222)
        self._create_top_view(ax2, user_mission, simulated_missions, conflicts)
        
        # Side View (2D)
        ax3 = fig.add_subplot(223)
        self._create_side_view(ax3, user_mission, simulated_missions, conflicts)
        
        # Conflict Summary
        ax4 = fig.add_subplot(224)
        self._create_conflict_summary(ax4, conflicts)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Analysis plot saved: {save_path}")
        
        plt.show()
    
    def _plot_mission(self, ax, mission, label, color, style, linewidth=2, markersize=6):
        """Plot a single mission in 3D"""
        waypoints = np.array(mission['waypoints'])
        
        if len(waypoints) == 0:
            return
        
        # Plot trajectory
        x, y, z = waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]
        ax.plot(x, y, z, style, color=color, label=label, linewidth=linewidth, markersize=markersize)
        
        # Add start and end markers
        ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=100, alpha=0.8)
        ax.scatter(x[-1], y[-1], z[-1], color=color, marker='s', s=100, alpha=0.8)
        
        # Add mission ID annotation
        ax.text(x[0], y[0], z[0], f' {label} Start', fontsize=8, color=color)
        ax.text(x[-1], y[-1], z[-1], f' {label} End', fontsize=8, color=color)
    
    def _highlight_conflicts(self, ax, conflicts):
        """Highlight conflict areas in the plot"""
        if not conflicts:
            return
        
        for i, conflict in enumerate(conflicts):
            if 'closest_points' in conflict:
                points = conflict['closest_points']
                
                # Extract points for user mission and conflicting mission
                user_point = None
                sim_point = None
                sim_mission_id = None
                
                for mission_id, point in points.items():
                    if mission_id == 'user_mission':
                        user_point = np.array(point)
                    else:
                        sim_point = np.array(point)
                        sim_mission_id = mission_id
                
                if user_point is not None and sim_point is not None:
                    # Draw line between conflict points
                    ax.plot([user_point[0], sim_point[0]], 
                           [user_point[1], sim_point[1]], 
                           [user_point[2], sim_point[2]], 
                           'r-', linewidth=3, alpha=0.8, label='Conflict' if i == 0 else "")
                    
                    # Mark conflict points
                    ax.scatter(*user_point, color='red', marker='X', s=200, alpha=0.9)
                    ax.scatter(*sim_point, color='red', marker='X', s=200, alpha=0.9)
                    
                    # Add conflict distance annotation
                    mid_point = (user_point + sim_point) / 2
                    distance = conflict['min_distance']
                    ax.text(mid_point[0], mid_point[1], mid_point[2], 
                           f'{distance:.1f}m', fontsize=9, color='red', weight='bold')
    
    def _add_safety_buffer_visualization(self, ax, user_mission):
        """Add safety buffer visualization around user mission"""
        waypoints = np.array(user_mission['waypoints'])
        
        if len(waypoints) == 0:
            return
        
        # Create buffer zones around key waypoints (start, middle, end)
        key_indices = [0, len(waypoints)//2, -1] if len(waypoints) > 2 else [0, -1]
        
        for idx in key_indices:
            point = waypoints[idx]
            
            # Create a transparent sphere for safety buffer
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 10)
            
            x = self.safety_buffer_m * np.outer(np.cos(u), np.sin(v)) + point[0]
            y = self.safety_buffer_m * np.outer(np.sin(u), np.sin(v)) + point[1]
            z = self.safety_buffer_m * np.outer(np.ones(np.size(u)), np.cos(v)) + point[2]
            
            ax.plot_surface(x, y, z, color='red', alpha=0.1, linewidth=0)
    
    def _customize_plot(self, ax, user_mission, simulated_missions):
        """Customize the 3D plot appearance"""
        # Set labels and title
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('🚀 Drone Mission Conflict Visualization\n(3D Space - Constant Speed Assumption)', 
                    fontsize=14, weight='bold')
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Set equal aspect ratio
        self._set_axes_equal(ax)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def _set_axes_equal(self, ax):
        """Set equal aspect ratio for 3D plots"""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    def _setup_animation_plot(self, ax, user_mission, simulated_missions):
        """Setup the base plot for animation"""
        # Plot all missions as background
        waypoints = np.array(user_mission['waypoints'])
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'b-', alpha=0.2, label='User Mission Path')
        
        for mission_id, mission in simulated_missions.items():
            waypoints = np.array(mission['waypoints'])
            ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'g-', alpha=0.2, label=f'{mission_id} Path')
        
        self._customize_plot(ax, user_mission, simulated_missions)
    
    def _update_animation(self, frame, ax):
        """Update animation frame with constant speed assumption"""
        # Clear previous drone positions
        for artist in ax.collections[:]:
            if hasattr(artist, '_is_drone_marker'):
                artist.remove()
        
        current_artists = []
        total_frames = 50
        
        # Update user mission drone position
        user_pos = self._get_position_at_progress(self.user_mission, frame / total_frames)
        user_marker = ax.scatter(*user_pos, color='blue', s=150, marker='o', alpha=0.9)
        user_marker._is_drone_marker = True
        current_artists.append(user_marker)
        
        # Update simulated missions drone positions
        colors = ['green', 'orange', 'purple', 'brown']
        for i, (mission_id, mission) in enumerate(self.simulated_missions.items()):
            sim_pos = self._get_position_at_progress(mission, frame / total_frames)
            color = colors[i % len(colors)]
            sim_marker = ax.scatter(*sim_pos, color=color, s=120, marker='^', alpha=0.8)
            sim_marker._is_drone_marker = True
            current_artists.append(sim_marker)
            
            # Add small label
            label = ax.text(sim_pos[0], sim_pos[1], sim_pos[2] + 5, mission_id, 
                           fontsize=7, color=color, ha='center')
            label._is_drone_marker = True
            current_artists.append(label)
        
        # Update progress in title
        progress = (frame / total_frames) * 100
        ax.set_title(f'🚀 Mission Progress: {progress:.0f}% (Constant Speed)\nRed X = Conflict Points', 
                    fontsize=12, weight='bold')
        
        return current_artists
    
    def _get_position_at_progress(self, mission, progress):
        """Get position along mission path at given progress (0-1) with constant speed"""
        waypoints = np.array(mission['waypoints'])
        
        if len(waypoints) == 1:
            return waypoints[0]
        
        # Calculate total path length
        segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        total_length = np.sum(segment_lengths)
        
        if total_length == 0:
            return waypoints[0]
        
        # Find position based on progress
        target_distance = progress * total_length
        current_distance = 0
        
        for i in range(len(segment_lengths)):
            if current_distance + segment_lengths[i] >= target_distance:
                # Position is in this segment
                segment_progress = (target_distance - current_distance) / segment_lengths[i]
                position = waypoints[i] + segment_progress * (waypoints[i+1] - waypoints[i])
                return position
            current_distance += segment_lengths[i]
        
        return waypoints[-1]
    
    def _create_3d_view(self, ax, user_mission, simulated_missions, conflicts):
        """Create 3D view for analysis plot"""
        # Plot user mission
        self._plot_mission(ax, user_mission, 'User Mission', 'blue', 'o-', linewidth=3, markersize=6)
        
        # Plot simulated missions
        colors = ['green', 'orange', 'purple', 'brown']
        for i, (mission_id, mission) in enumerate(simulated_missions.items()):
            color = colors[i % len(colors)]
            self._plot_mission(ax, mission, mission_id, color, '^-', linewidth=2, markersize=4)
        
        # Highlight conflicts
        self._highlight_conflicts(ax, conflicts)
        
        ax.set_title('3D Mission View')
        ax.grid(True, alpha=0.3)
    
    def _create_top_view(self, ax, user_mission, simulated_missions, conflicts):
        """Create top-down 2D view"""
        # Plot user mission (x,y only)
        waypoints = np.array(user_mission['waypoints'])
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'bo-', linewidth=2, markersize=4, label='User Mission')
        
        # Plot simulated missions
        colors = ['green', 'orange', 'purple', 'brown']
        for i, (mission_id, mission) in enumerate(simulated_missions.items()):
            waypoints = np.array(mission['waypoints'])
            color = colors[i % len(colors)]
            ax.plot(waypoints[:, 0], waypoints[:, 1], color + '^-', linewidth=1, markersize=3, label=mission_id)
        
        # Highlight conflicts
        for conflict in conflicts:
            if 'closest_points' in conflict:
                points = conflict['closest_points']
                user_point = points.get('user_mission')
                sim_point = None
                for mission_id, point in points.items():
                    if mission_id != 'user_mission':
                        sim_point = point
                        break
                
                if user_point and sim_point:
                    ax.plot([user_point[0], sim_point[0]], 
                           [user_point[1], sim_point[1]], 'r-', linewidth=2, alpha=0.8)
                    ax.scatter(user_point[0], user_point[1], color='red', marker='X', s=100)
                    ax.scatter(sim_point[0], sim_point[1], color='red', marker='X', s=100)
        
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title('Top View (X-Y Plane)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _create_side_view(self, ax, user_mission, simulated_missions, conflicts):
        """Create side elevation view (x,z)"""
        # Plot user mission (x,z only)
        waypoints = np.array(user_mission['waypoints'])
        ax.plot(waypoints[:, 0], waypoints[:, 2], 'bo-', linewidth=2, markersize=4, label='User Mission')
        
        # Plot simulated missions
        colors = ['green', 'orange', 'purple', 'brown']
        for i, (mission_id, mission) in enumerate(simulated_missions.items()):
            waypoints = np.array(mission['waypoints'])
            color = colors[i % len(colors)]
            ax.plot(waypoints[:, 0], waypoints[:, 2], color + '^-', linewidth=1, markersize=3, label=mission_id)
        
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Side View (X-Z Plane)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_conflict_summary(self, ax, conflicts):
        """Create conflict summary chart"""
        if not conflicts:
            ax.text(0.5, 0.5, '✅ No Conflicts Detected\nAll missions are safe!', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax.set_title('Conflict Summary')
            ax.axis('off')
            return
        
        # Count conflicts by severity
        severities = [c['severity'] for c in conflicts]
        severity_counts = {sev: severities.count(sev) for sev in set(severities)}
        
        colors = {'critical': 'red', 'high': 'orange', 'moderate': 'yellow'}
        labels = [f'{sev.capitalize()} ({count})' for sev, count in severity_counts.items()]
        sizes = list(severity_counts.values())
        color_list = [colors.get(sev, 'gray') for sev in severity_counts.keys()]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=color_list, autopct='%1.0f%%',
                                         startangle=90)
        
        # Style the chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f'Conflict Severity\nTotal: {len(conflicts)} conflicts', fontweight='bold')
        
        # Add summary text
        min_distance = min([c['min_distance'] for c in conflicts])
        ax.text(0.5, -0.1, f'Closest approach: {min_distance:.1f}m', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')


# Utility function for quick visualization
def quick_visualize(user_mission, simulated_missions, conflicts=None, plot_type='static'):
    """
    Quick visualization utility for missions
    
    Parameters:
    - plot_type: 'static', 'animated', or 'analysis'
    """
    visualizer = MissionVisualizer()
    
    if plot_type == 'static':
        visualizer.create_static_3d_plot(user_mission, simulated_missions, conflicts or [])
    elif plot_type == 'animated':
        visualizer.create_animated_3d_plot(user_mission, simulated_missions, conflicts or [])
    elif plot_type == 'analysis':
        visualizer.create_conflict_analysis_plot(user_mission, simulated_missions, conflicts or [])


# Simple demo
if __name__ == "__main__":
    from conflict_detector import create_sample_mission
    from datetime import datetime
    
    # Create sample data for testing
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    user_mission = create_sample_mission(
        "user_demo",
        [[0, 0, 100], [100, 50, 110], [200, 100, 120]],
        base_time,
        base_time + pd.Timedelta(minutes=30)
    )
    
    simulated_missions = {
        "mission_001": create_sample_mission(
            "mission_001",
            [[5, 5, 102], [105, 55, 108], [205, 105, 118]],
            base_time + pd.Timedelta(minutes=5),
            base_time + pd.Timedelta(minutes=25)
        )
    }
    
    # Create demo conflict
    detector = ConflictDetector()
    conflicts = [detector.detect_conflict(user_mission, simulated_missions["mission_001"])]
    
    print("🎨 Generating demo visualization...")
    quick_visualize(user_mission, simulated_missions, conflicts, plot_type='static')