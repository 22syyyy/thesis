import numpy as np
import pandas as pd
import math

class TrajectoryGenerator:

    def __init__(self, initial_pos):
        
        self.init_x, self.init_y, self.init_z = initial_pos
        
        self.radius = 0.3
        self.base_height = self.init_z
        self.z_amplitude = 0.1
        self.num_points = 200
        self.num_turns = 2
        self.lookahead_points = 3  
        self.reached_goal = False
        self.goal_threshold = 0.2  
        self.trajectory = self.generate_trajectory()

    def get_position(self, current_pos):

        if self.reached_goal:
            last_point = self.trajectory.iloc[-1]
            return [last_point['ref.x'], last_point['ref.y'], last_point['ref.z']], last_point['ref.yaw']

        x, y, z = current_pos
        traj_xyz = self.trajectory[['ref.x', 'ref.y', 'ref.z']].values

        distances = np.linalg.norm(traj_xyz - np.array([x, y, z]), axis=1)
        min_index = np.argmin(distances)

        target_index = min(min_index + self.lookahead_points, len(self.trajectory) - 1)

        final_point = traj_xyz[-1]
        dist_to_goal = np.linalg.norm(np.array([x, y, z]) - final_point)
        if dist_to_goal < self.goal_threshold:
            self.reached_goal = True

        point = self.trajectory.iloc[target_index]
        return [point['ref.x'], point['ref.y'], point['ref.z']], point['ref.yaw']

    def generate_trajectory(self):
        # Generate angular positions for multiple turns
        angles = np.linspace(0, 2 * np.pi * self.num_turns, self.num_points)

        # Position
        x = self.init_x + self.radius * np.cos(angles)
        y = self.init_y + self.radius * np.sin(angles)
        z = self.init_z - self.z_amplitude * np.sin(2 * angles)  # Oscillate in z

        # Tangential direction for yaw
        dx = np.roll(x, -1) - x
        dy = np.roll(y, -1) - y
        yaw = np.arctan2(dy, dx)
        yaw[-1] = yaw[-2]  # Fix wraparound

        # Combine into DataFrame
        trajectory = pd.DataFrame({
            'ref.x': x,
            'ref.y': y,
            'ref.z': z,
            'ref.yaw': yaw
        })

        return trajectory
