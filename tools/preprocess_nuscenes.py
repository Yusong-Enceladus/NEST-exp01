import os
import json
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import pickle
from pathlib import Path

class NuScenesPreprocessor:
    def __init__(self, data_root: str, version: str = "v1.0-mini"):
        """
        Initialize NuScenes data preprocessor
        Args:
            data_root: Path to nuScenes dataset
            version: Dataset version
        """
        self.data_root = Path(data_root)
        self.version = version
        self.data_path = self.data_root / version
        
        # Load meta data
        self.load_meta_data()
        
    def load_meta_data(self):
        """Load all meta data from json files"""
        print("Loading meta data...")
        
        try:
            # Load all json files
            self.samples = self.load_json("sample.json")
            self.sample_anns = self.load_json("sample_annotation.json")
            self.scenes = self.load_json("scene.json")
            self.sample_data = self.load_json("sample_data.json")
            
            # Print sample data structure for debugging
            if len(self.sample_anns) > 0:
                print("\nSample annotation structure:")
                for key in self.sample_anns[0].keys():
                    print(f"- {key}")
            
            # Create helper dictionaries for faster lookup
            self.sample_lookup = {s["token"]: s for s in self.samples}
            self.scene_lookup = {s["token"]: s for s in self.scenes}
            self.ann_lookup = {a["token"]: a for a in self.sample_anns}
            
            print(f"\nMeta data loaded successfully:")
            print(f"- {len(self.samples)} samples")
            print(f"- {len(self.sample_anns)} annotations")
            print(f"- {len(self.scenes)} scenes")
            print(f"- {len(self.sample_data)} sample data entries")
        except Exception as e:
            print(f"Error loading meta data: {str(e)}")
            raise
    
    def load_json(self, filename: str) -> List[Dict]:
        """Load json file"""
        try:
            with open(self.data_path / filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {filename} not found in {self.data_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: {filename} is not a valid JSON file")
            raise
    
    def extract_trajectories(self) -> Dict:
        """Extract trajectories for all agents"""
        print("Extracting trajectories...")
        trajectories = {}
        
        # Process each scene
        for scene in tqdm(self.scenes):
            scene_token = scene["token"]
            current_sample = self.sample_lookup[scene["first_sample_token"]]
            
            while current_sample is not None:
                # Get all annotations for current sample
                sample_anns = [ann for ann in self.sample_anns if ann["sample_token"] == current_sample["token"]]
                
                # Process each annotation
                for ann in sample_anns:
                    instance_token = ann["instance_token"]
                    if instance_token not in trajectories:
                        trajectories[instance_token] = []
                    
                    # Add position and velocity
                    # Print first annotation for debugging
                    if len(trajectories[instance_token]) == 0:
                        print("\nFirst annotation example:")
                        print(json.dumps(ann, indent=2))
                    
                    trajectory_point = {
                        "timestamp": current_sample["timestamp"],
                        "translation": ann["translation"],
                        "size": ann.get("size", [0, 0, 0]),
                        "rotation": ann.get("rotation", [0, 0, 0, 1]),
                        "velocity": ann.get("velocity", [0, 0, 0]),
                        "acceleration": ann.get("acceleration", [0, 0, 0]),
                        "category": ann.get("category_name", ann.get("category", "unknown"))
                    }
                    trajectories[instance_token].append(trajectory_point)
                
                # Move to next sample
                if "next" not in current_sample or current_sample["next"] == "":
                    current_sample = None
                else:
                    current_sample = self.sample_lookup[current_sample["next"]]
        
        # Interpolate trajectories that are too short
        print("\nChecking trajectory lengths...")
        min_frames = 50  # 20 frames observation + 30 frames prediction
        interpolated_trajectories = {}
        skipped = 0
        interpolated = 0
        sufficient = 0
        
        for instance_token, traj in trajectories.items():
            if len(traj) < 2:  # Skip trajectories with less than 2 points
                skipped += 1
                continue
            
            # Sort by timestamp
            traj = sorted(traj, key=lambda x: x["timestamp"])
            
            if len(traj) >= min_frames:
                # If trajectory is long enough, keep as is
                interpolated_trajectories[instance_token] = traj
                sufficient += 1
                continue
            
            # Calculate required interpolation factor
            current_frames = len(traj)
            required_factor = np.ceil(min_frames / current_frames)
            
            # Prepare interpolation points
            timestamps = np.array([p["timestamp"] for p in traj])
            positions = np.array([p["translation"] for p in traj])
            velocities = np.array([p["velocity"] for p in traj])
            
            # Create interpolation timestamps
            t_min, t_max = timestamps[0], timestamps[-1]
            num_points = int(current_frames * required_factor)  # Just enough points to meet minimum
            new_timestamps = np.linspace(t_min, t_max, num_points)
            
            # Interpolate positions and velocities
            interp_positions = np.array([
                np.interp(new_timestamps, timestamps, positions[:, i])
                for i in range(3)
            ]).T
            
            interp_velocities = np.array([
                np.interp(new_timestamps, timestamps, velocities[:, i])
                for i in range(3)
            ]).T
            
            # Create interpolated trajectory points
            interpolated_traj = []
            for t, pos, vel in zip(new_timestamps, interp_positions, interp_velocities):
                point = {
                    "timestamp": int(t),
                    "translation": pos.tolist(),
                    "velocity": vel.tolist(),
                    "acceleration": [0, 0, 0],  # Set to zero for interpolated points
                    "size": traj[0]["size"],  # Use first point's size
                    "rotation": traj[0]["rotation"],  # Use first point's rotation
                    "category": traj[0]["category"]
                }
                interpolated_traj.append(point)
            
            interpolated_trajectories[instance_token] = interpolated_traj
            interpolated += 1
        
        print(f"\nTrajectory processing summary:")
        print(f"- Total trajectories: {len(trajectories)}")
        print(f"- Sufficient length: {sufficient}")
        print(f"- Interpolated: {interpolated}")
        print(f"- Skipped (too short): {skipped}")
        print(f"- Final trajectories: {len(interpolated_trajectories)}")
        
        return interpolated_trajectories
    
    def build_interaction_hypergraph(self, trajectories: Dict) -> Dict:
        """Build interaction hypergraph based on spatial-temporal proximity"""
        print("Building interaction hypergraph...")
        hypergraph = {}
        
        # Group trajectories by timestamp
        timestamp_groups = {}
        for instance_token, traj in trajectories.items():
            for frame in traj:
                ts = frame["timestamp"]
                if ts not in timestamp_groups:
                    timestamp_groups[ts] = []
                timestamp_groups[ts].append((instance_token, frame))
        
        # Build hyperedges based on spatial proximity
        for ts, instances in timestamp_groups.items():
            hypergraph[ts] = []
            positions = np.array([inst[1]["translation"] for inst in instances])
            
            # Use distance-based clustering to form hyperedges
            for i, (inst_token, frame) in enumerate(instances):
                # Find nearby instances within 50m radius
                dists = np.linalg.norm(positions - positions[i], axis=1)
                nearby_indices = np.where(dists < 50.0)[0]
                
                if len(nearby_indices) > 1:  # Only create hyperedge if there are interactions
                    hyperedge = [instances[idx][0] for idx in nearby_indices]
                    hypergraph[ts].append(hyperedge)
        
        print(f"Built hypergraph with {sum(len(edges) for edges in hypergraph.values())} total hyperedges")
        return hypergraph
    
    def process_dataset(self, output_path: str):
        """Process the entire dataset and save results"""
        try:
            # Extract trajectories
            trajectories = self.extract_trajectories()
            
            # Build interaction hypergraph
            hypergraph = self.build_interaction_hypergraph(trajectories)
            
            # Save processed data
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as both pickle and numpy formats for compatibility
            with open(output_dir / "trajectories.pkl", "wb") as f:
                pickle.dump(trajectories, f)
            
            with open(output_dir / "hypergraph.pkl", "wb") as f:
                pickle.dump(hypergraph, f)
            
            # Also save as numpy arrays for easier loading
            trajectory_data = {
                "instance_tokens": list(trajectories.keys()),
                "trajectories": [trajectories[k] for k in trajectories.keys()]
            }
            np.savez(
                output_dir / "processed_data.npz",
                trajectories=trajectory_data,
                hypergraph=hypergraph
            )
            
            print(f"Processed data saved to {output_path}")
            print(f"- Number of trajectories: {len(trajectories)}")
            print(f"- Number of timestamps: {len(hypergraph)}")
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            raise

def main():
    try:
        # Initialize preprocessor
        preprocessor = NuScenesPreprocessor(
            data_root="data/nuscenes",
            version="v1.0-mini"
        )
        
        # Process dataset
        preprocessor.process_dataset("data/processed/nuscenes-mini")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 