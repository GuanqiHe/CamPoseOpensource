import numpy as np
import os
import json

# Global camera names; one pose will be generated per name for each item
camera_names = ['cam0', 'cam1']

def generate_camera_poses(num_cameras=100, 
                         workspace_center=np.array([0.0, 0.0, 0.8]),
                         min_distance=0.7, max_distance=1.2,
                         min_elevation=30, max_elevation=60,
                         azimuth_range=90,
                         cube_size=0.3,
                         output_file="camera_poses.json", seed=42):
    """
    Generate camera poses around the workspace and save to file.
    Each pose item contains one pose per camera.
    Hardcoded to a single camera ['cam0'].
    
    Args:
        num_cameras (int): Number of camera poses to generate
        workspace_center (np.array): Center of the workspace
        min_distance (float): Minimum distance from workspace center
        max_distance (float): Maximum distance from workspace center
        min_elevation (float): Minimum elevation angle in degrees
        max_elevation (float): Maximum elevation angle in degrees
        azimuth_range (float): Azimuth range in degrees (centered around front view)
        output_file (str): Output JSON file path
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    def sample_pose(center, min_dist, max_dist, min_elev, max_elev, azimuth_deg):
        # Center azimuth around π/2 (90 degrees, facing +Y direction)
        # Sample within ±azimuth_range/2 from center
        azimuth_center = 0
        azimuth_half_range = np.radians(azimuth_deg / 2)
        azimuth = np.random.uniform(azimuth_center - azimuth_half_range, 
                                   azimuth_center + azimuth_half_range)
        elevation = np.random.uniform(np.radians(min_elev), np.radians(max_elev))
        distance = np.random.uniform(min_dist, max_dist)
        
        x = center[0] + distance * np.cos(elevation) * np.cos(azimuth)
        y = center[1] + distance * np.cos(elevation) * np.sin(azimuth)
        z = center[2] + distance * np.sin(elevation)
        
        camera_pos = np.array([x, y, z])
        
        # Randomize the center point uniformly within a small cube
        center_offset = np.random.uniform(-cube_size/2, cube_size/2, 3)
        randomized_center = center + center_offset
        
        forward = randomized_center - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 0, 1])
        up = up / np.linalg.norm(up)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        cam_to_world = np.eye(4)
        cam_to_world[:3, 0] = right
        cam_to_world[:3, 1] = up
        cam_to_world[:3, 2] = -forward
        cam_to_world[:3, 3] = camera_pos
        
        return cam_to_world.tolist()
    
    names = ['cam0']
    poses = []
    for _ in range(num_cameras):
        item_poses = []
        for _cam in names:
            pose = sample_pose(workspace_center, min_distance, max_distance, 
                               min_elevation, max_elevation, azimuth_range)
            item_poses.append(pose)
        poses.append(item_poses)
    
    data = {
        "config": {
        "num_cameras": num_cameras,
        "camera_names": names,
        "workspace_center": workspace_center.tolist(),
        "center_rand_cube_size": cube_size,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "min_elevation": min_elevation,
        "max_elevation": max_elevation,
        "azimuth_range": azimuth_range,
        "seed": seed
        },
        "poses": poses,
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {num_cameras} camera poses")
    print(f"Saved to {output_file}")

def generate_camera_poses_intervals(num_cameras=100,
                                   workspace_center=np.array([0.0, 0.0, 0.8]),
                                   min_distance=0.7, max_distance=1.2,
                                   min_elevation=30, max_elevation=60,
                                   total_range=360, interval_length=30, start_azimuth=-180,
                                   cube_size=0.3,
                                   output_file="camera_poses_intervals.json", seed=42):
    """
    Generate camera poses around the workspace on specific azimuth intervals and save to file.
    Each pose item contains one pose per camera in `camera_names`.
    
    Args:
        num_cameras (int): Number of camera poses to generate
        workspace_center (np.array): Center of the workspace
        min_distance (float): Minimum distance from workspace center
        max_distance (float): Maximum distance from workspace center
        min_elevation (float): Minimum elevation angle in degrees
        max_elevation (float): Maximum elevation angle in degrees
        total_range (float): Total azimuth range in degrees
        interval_length (float): Length of each interval in degrees
        start_azimuth (float): Starting azimuth angle in degrees
        cube_size (float): Size of cube for center randomization
        output_file (str): Output JSON file path
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Calculate intervals
    num_intervals = int(total_range / (2 * interval_length))
    intervals = []
    
    for i in range(num_intervals):
        interval_start = start_azimuth + i * 2 * interval_length
        interval_end = interval_start + interval_length
        intervals.append((interval_start, interval_end))
    
    def sample_pose(center, min_dist, max_dist, min_elev, max_elev, intervals):
        # Randomly select an interval
        interval = intervals[np.random.randint(len(intervals))]
        interval_start, interval_end = interval
        
        # Sample azimuth uniformly within the selected interval
        azimuth = np.random.uniform(np.radians(interval_start), np.radians(interval_end))
        elevation = np.random.uniform(np.radians(min_elev), np.radians(max_elev))
        distance = np.random.uniform(min_dist, max_dist)
        
        x = center[0] + distance * np.cos(elevation) * np.cos(azimuth)
        y = center[1] + distance * np.cos(elevation) * np.sin(azimuth)
        z = center[2] + distance * np.sin(elevation)
        
        camera_pos = np.array([x, y, z])
        
        # Randomize the center point uniformly within a small cube
        center_offset = np.random.uniform(-cube_size/2, cube_size/2, 3)
        randomized_center = center + center_offset
        
        forward = randomized_center - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 0, 1])
        up = up / np.linalg.norm(up)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        cam_to_world = np.eye(4)
        cam_to_world[:3, 0] = right
        cam_to_world[:3, 1] = up
        cam_to_world[:3, 2] = -forward
        cam_to_world[:3, 3] = camera_pos
        
        return cam_to_world.tolist()
    
    poses = []
    for _ in range(num_cameras):
        item_poses = []
        for _cam in camera_names:
            pose = sample_pose(workspace_center, min_distance, max_distance, 
                               min_elevation, max_elevation, intervals)
            item_poses.append(pose)
        poses.append(item_poses)
    
    data = {
        "config": {
            "num_cameras": num_cameras,
            "workspace_center": workspace_center.tolist(),
            "camera_names": camera_names,
            "center_rand_cube_size": cube_size,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "min_elevation": min_elevation,
            "max_elevation": max_elevation,
            "total_range": total_range,
            "interval_length": interval_length,
            "start_azimuth": start_azimuth,
            "intervals": intervals,
            "seed": seed
        },
        "poses": poses,
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {num_cameras} camera poses on {len(intervals)} intervals")
    print(f"Intervals: {intervals}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    generate_camera_poses_intervals(num_cameras=500,
                                    workspace_center=np.array([0.0, 0.0, 0.8]),
                                    min_distance=0.7, max_distance=1.2,
                                    min_elevation=30, max_elevation=60,
                                    total_range=240, interval_length=30, start_azimuth=-90,
                                    cube_size=0.3,
                                    output_file="interval_test_cameras.json", seed=42)