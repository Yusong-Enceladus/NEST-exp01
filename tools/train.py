import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import math

class NuScenesDataset(Dataset):
    def __init__(self, data_path: str, obs_len: int = 20, pred_len: int = 30, distance_threshold: float = 50.0,
                 augment: bool = True, max_neighbors: int = 20):
        """
        Args:
            data_path: Path to processed data directory
            obs_len: Number of observed timesteps
            pred_len: Number of timesteps to predict
            distance_threshold: Maximum distance (in meters) to consider for spatial proximity
            augment: Whether to use data augmentation during training
            max_neighbors: Maximum number of neighbors to consider
        """
        self.data_path = Path(data_path)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.distance_threshold = distance_threshold
        self.augment = augment
        self.max_neighbors = max_neighbors
        
        print(f"\nInitializing dataset from {self.data_path}")
        print(f"Observation length: {obs_len}, Prediction length: {pred_len}")
        print(f"Distance threshold: {distance_threshold} meters")
        print(f"Data augmentation: {augment}")
        print(f"Maximum neighbors: {max_neighbors}")
        
        # Load processed data
        try:
            with open(self.data_path / "trajectories.pkl", "rb") as f:
                self.trajectories = pickle.load(f)
            with open(self.data_path / "hypergraph.pkl", "rb") as f:
                self.hypergraph = pickle.load(f)
            
            print(f"Loaded {len(self.trajectories)} trajectories")
            print(f"Loaded hypergraph with {len(self.hypergraph)} timestamps")
            
            # Compute normalization statistics
            self.compute_normalization_stats()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
        # Convert trajectories to sequences
        self.prepare_sequences()
    
    def compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        print("\nComputing normalization statistics...")
        
        all_positions = []
        all_velocities = []
        
        for trajectory in self.trajectories.values():
            positions = np.array([frame["translation"] for frame in trajectory])
            velocities = np.array([frame["velocity"] for frame in trajectory])
            
            all_positions.append(positions)
            all_velocities.append(velocities)
        
        all_positions = np.concatenate(all_positions, axis=0)
        all_velocities = np.concatenate(all_velocities, axis=0)
        
        # Compute statistics
        self.pos_mean = np.mean(all_positions, axis=0)
        self.pos_std = np.std(all_positions, axis=0)
        self.vel_mean = np.mean(all_velocities, axis=0)
        self.vel_std = np.std(all_velocities, axis=0)
        
        # Avoid division by zero
        self.pos_std = np.where(self.pos_std < 1e-6, 1.0, self.pos_std)
        self.vel_std = np.where(self.vel_std < 1e-6, 1.0, self.vel_std)
        
        print("Normalization statistics:")
        print(f"Position mean: {self.pos_mean}")
        print(f"Position std: {self.pos_std}")
        print(f"Velocity mean: {self.vel_mean}")
        print(f"Velocity std: {self.vel_std}")
        
    def augment_sequence(self, sequence):
        """Apply data augmentation to a sequence"""
        if not self.augment:
            return sequence
        
        # Random rotation
        angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to positions
        sequence["obs_pos"] = np.dot(sequence["obs_pos"], rot_matrix)
        sequence["pred_pos"] = np.dot(sequence["pred_pos"], rot_matrix)
        
        # Apply rotation to velocities
        sequence["obs_vel"] = np.dot(sequence["obs_vel"], rot_matrix)
        sequence["pred_vel"] = np.dot(sequence["pred_vel"], rot_matrix)
        
        # Apply rotation to accelerations
        sequence["obs_acc"] = np.dot(sequence["obs_acc"], rot_matrix)
        sequence["pred_acc"] = np.dot(sequence["pred_acc"], rot_matrix)
        
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        sequence["obs_pos"] *= scale
        sequence["pred_pos"] *= scale
        sequence["obs_vel"] *= scale
        sequence["pred_vel"] *= scale
        sequence["obs_acc"] *= scale
        sequence["pred_acc"] *= scale
        
        # Random noise
        noise_std = 0.01
        sequence["obs_pos"] += np.random.normal(0, noise_std, sequence["obs_pos"].shape)
        sequence["obs_vel"] += np.random.normal(0, noise_std, sequence["obs_vel"].shape)
        sequence["obs_acc"] += np.random.normal(0, noise_std, sequence["obs_acc"].shape)
        
        return sequence
    
    def prepare_sequences(self):
        """Convert trajectories to sequence format with filtering and augmentation"""
        self.sequences = []
        skipped_short = 0
        skipped_static = 0
        total_trajectories = len(self.trajectories)
        
        print("\nPreparing sequences...")
        for instance_token, trajectory in tqdm(self.trajectories.items(), desc="Processing trajectories"):
            # Sort trajectory by timestamp
            trajectory = sorted(trajectory, key=lambda x: x["timestamp"])
            
            # Skip if trajectory is too short
            min_length = self.obs_len + self.pred_len
            if len(trajectory) < min_length:
                skipped_short += 1
                continue
            
            try:
                # Convert to numpy arrays
                positions = np.array([frame["translation"] for frame in trajectory])
                velocities = np.array([frame["velocity"] for frame in trajectory])
                accelerations = np.array([frame["acceleration"] for frame in trajectory])
                
                # Skip static or near-static trajectories
                total_displacement = np.linalg.norm(positions[-1] - positions[0])
                if total_displacement < 1.0:  # Less than 1 meter movement
                    skipped_static += 1
                    continue
                
                # Normalize data
                positions = (positions - self.pos_mean) / self.pos_std
                velocities = (velocities - self.vel_mean) / self.vel_std
                
                # Create sequences with sliding window
                for i in range(len(trajectory) - self.obs_len - self.pred_len + 1):
                    sequence = {
                        "instance_token": instance_token,
                        "obs_pos": positions[i:i+self.obs_len],
                        "obs_vel": velocities[i:i+self.obs_len],
                        "obs_acc": accelerations[i:i+self.obs_len],
                        "pred_pos": positions[i+self.obs_len:i+self.obs_len+self.pred_len],
                        "pred_vel": velocities[i+self.obs_len:i+self.obs_len+self.pred_len],
                        "pred_acc": accelerations[i+self.obs_len:i+self.obs_len+self.pred_len],
                        "timestamp": trajectory[i+self.obs_len-1]["timestamp"]
                    }
                    
                    # Apply data augmentation
                    sequence = self.augment_sequence(sequence)
                    
                    self.sequences.append(sequence)
            except Exception as e:
                print(f"\nError processing trajectory {instance_token}: {str(e)}")
                continue
        
        print(f"\nSequence preparation complete:")
        print(f"- Total trajectories: {total_trajectories}")
        print(f"- Skipped (too short): {skipped_short}")
        print(f"- Skipped (static): {skipped_static}")
        print(f"- Generated sequences: {len(self.sequences)}")
        
        if len(self.sequences) == 0:
            raise ValueError("No valid sequences generated. Please check the data format and minimum sequence length requirements.")
    
    def __len__(self):
        return len(self.sequences)
    
    def get_spatial_neighbors(self, target_pos, all_positions, timestamp):
        """Get neighbors based on spatial proximity
        Args:
            target_pos: Target agent position [3]
            all_positions: All agents' positions at current timestamp [N, 3]
            timestamp: Current timestamp
        Returns:
            neighbor_indices: Indices of neighbors within distance threshold
        """
        # Compute distances to all agents
        distances = np.linalg.norm(all_positions - target_pos, axis=1)
        
        # Find agents within threshold (excluding self)
        neighbor_indices = np.where(
            (distances < self.distance_threshold) & (distances > 0)
        )[0]
        
        return neighbor_indices
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Get hypergraph for the timestamp
        timestamp = sequence["timestamp"]
        hyperedges = self.hypergraph.get(timestamp, [])
        
        # Get all agents' positions at current timestamp
        current_positions = {}
        current_velocities = {}
        
        for token, traj in self.trajectories.items():
            frames = [f for f in traj if abs(f["timestamp"] - timestamp) < 1e6]
            if frames:
                frame = frames[-1]  # Use the latest frame within window
                pos = (np.array(frame["translation"]) - self.pos_mean) / self.pos_std
                vel = (np.array(frame["velocity"]) - self.vel_mean) / self.vel_std
                current_positions[token] = pos.copy()
                current_velocities[token] = vel.copy()
        
        tokens = list(current_positions.keys())
        positions = np.stack([current_positions[t] for t in tokens])
        
        target_token = sequence["instance_token"]
        target_idx = tokens.index(target_token)
        target_pos = positions[target_idx].copy()
        
        neighbor_indices = self.get_spatial_neighbors(target_pos, positions, timestamp)
        
        if len(neighbor_indices) > self.max_neighbors:
            distances = np.linalg.norm(positions[neighbor_indices] - target_pos, axis=1)
            closest_indices = np.argsort(distances)[:self.max_neighbors]
            neighbor_indices = neighbor_indices[closest_indices]
        
        neighbor_pos_list = []
        neighbor_vel_list = []
        neighbor_tokens = []
        
        for idx_neighbor in neighbor_indices:
            token = tokens[idx_neighbor]
            traj = self.trajectories[token]
            frames = [f for f in traj if abs(f["timestamp"] - timestamp) < 1e6]
            if len(frames) >= self.obs_len:
                pos = np.array([f["translation"] for f in frames[-self.obs_len:]])
                vel = np.array([f["velocity"] for f in frames[-self.obs_len:]])
                pos = (pos - self.pos_mean) / self.pos_std
                vel = (vel - self.vel_mean) / self.vel_std
                neighbor_pos_list.append(pos.copy())
                neighbor_vel_list.append(vel.copy())
                neighbor_tokens.append(token)
        
        num_neighbors = len(neighbor_pos_list)
        if num_neighbors == 0:
            neighbor_pos = np.zeros((self.max_neighbors, self.obs_len, 3))
            neighbor_vel = np.zeros((self.max_neighbors, self.obs_len, 3))
            neighbor_mask = torch.zeros(self.max_neighbors, dtype=torch.bool)
            num_neighbors = 0
        else:
            neighbor_pos = np.stack(neighbor_pos_list, axis=0)
            neighbor_vel = np.stack(neighbor_vel_list, axis=0)
            if num_neighbors < self.max_neighbors:
                pad_size = self.max_neighbors - num_neighbors
                neighbor_pos = np.pad(neighbor_pos, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
                neighbor_vel = np.pad(neighbor_vel, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
                neighbor_mask = torch.cat([
                    torch.ones(num_neighbors, dtype=torch.bool),
                    torch.zeros(pad_size, dtype=torch.bool)
                ])
            else:
                neighbor_pos = neighbor_pos[:self.max_neighbors]
                neighbor_vel = neighbor_vel[:self.max_neighbors]
                neighbor_mask = torch.ones(self.max_neighbors, dtype=torch.bool)
        
        return {
            "obs_pos": torch.FloatTensor(sequence["obs_pos"].copy()),
            "obs_vel": torch.FloatTensor(sequence["obs_vel"].copy()),
            "obs_acc": torch.FloatTensor(sequence["obs_acc"].copy()),
            "pred_pos": torch.FloatTensor(sequence["pred_pos"].copy()),
            "pred_vel": torch.FloatTensor(sequence["pred_vel"].copy()),
            "pred_acc": torch.FloatTensor(sequence["pred_acc"].copy()),
            "neighbor_pos": torch.FloatTensor(neighbor_pos.copy()),
            "neighbor_vel": torch.FloatTensor(neighbor_vel.copy()),
            "neighbor_mask": neighbor_mask.clone(),
            "num_neighbors": num_neighbors,
            "lane": torch.FloatTensor(np.zeros((10, 64)))
        }

class NEST(nn.Module):
    def __init__(self, obs_len: int = 20, pred_len: int = 30, hidden_size: int = 256, num_modes: int = 6, num_heads: int = 8):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_modes = num_modes
        self.num_heads = num_heads
        
        # Neuromodulator components
        self.threshold_neuromodulator = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.connection_neuromodulator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Trajectory encoder
        self.encoder = nn.LSTM(
            input_size=9,  # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z]
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Neighbor encoder
        self.neighbor_encoder = nn.LSTM(
            input_size=6,  # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Hypergraph components
        self.vertex_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.hyperedge_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Multi-head attention for neighbor interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Mode prediction
        self.mode_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_modes)
        )
        
        # Trajectory decoder for each mode
        self.decoders = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            ) for _ in range(num_modes)
        ])
        
        # Output heads for each mode
        self.pos_predictors = nn.ModuleList([
            nn.Linear(hidden_size, 3) for _ in range(num_modes)
        ])
        self.vel_predictors = nn.ModuleList([
            nn.Linear(hidden_size, 3) for _ in range(num_modes)
        ])
        
        # Added Context Fusion modules per paper
        self.lane_encoder = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU()
        )
    
    def compute_clustering_coefficient(self, features):
        """Compute clustering coefficient matrix based on feature similarity, with eps to avoid division by zero"""
        # Compute pairwise similarity with eps to ensure numerical stability
        norm_features = F.normalize(features, p=2, dim=-1, eps=1e-6)
        similarity = torch.matmul(norm_features, norm_features.transpose(-2, -1))
        return similarity
    
    def build_small_world_hypergraph(self, features, neighbor_mask):
        """Build small-world hypergraph using Newman-Watts model"""
        batch_size = features.size(0)
        num_nodes = features.size(1)
        
        # Compute clustering coefficient
        C = self.compute_clustering_coefficient(features)  # [batch, num_nodes, num_nodes]
        
        # Get threshold from neuromodulator
        threshold = self.threshold_neuromodulator(C.mean(dim=-1, keepdim=True))  # [batch, num_nodes, 1]
        threshold = torch.clamp(threshold, min=1e-3, max=1.0 - 1e-3)
        
        # Initial connections based on threshold
        connections = (C >= threshold).float()
        
        # Get connection probability from neuromodulator
        beta = self.connection_neuromodulator(features.mean(dim=1, keepdim=True))  # [batch, 1, 1]
        beta = torch.clamp(beta, min=1e-3, max=1.0 - 1e-3)
        
        # Add random connections based on beta
        random_connections = (torch.rand_like(C) < beta).float()
        connections = connections + (1 - connections) * random_connections
        
        # Apply neighbor mask
        connections = connections * neighbor_mask.unsqueeze(1).float()
        
        return connections
    
    def hypergraph_message_passing(self, vertex_features, connections):
        """Perform message passing on hypergraph"""
        # Vertex to hyperedge
        hyperedge_features = torch.matmul(connections.transpose(-2, -1), vertex_features)
        hyperedge_features = self.hyperedge_encoder(hyperedge_features)
        
        # Hyperedge to vertex
        updated_features = torch.matmul(connections, hyperedge_features)
        updated_features = self.vertex_encoder(updated_features)
        
        return updated_features
    
    def forward(self, obs_pos, obs_vel, obs_acc, neighbor_pos, neighbor_vel, neighbor_mask, lane):
        batch_size = obs_pos.size(0)
        
        # Encode observed trajectory
        obs_features = torch.cat([obs_pos, obs_vel, obs_acc], dim=-1)  # [batch, obs_len, 9]
        obs_encoded, (h_n, c_n) = self.encoder(obs_features)
        
        # Encode neighbor trajectories
        neighbor_features = torch.cat([neighbor_pos, neighbor_vel], dim=-1)  # [batch, max_neighbors, obs_len, 6]
        B, N, T, _ = neighbor_features.size()
        neighbor_features = neighbor_features.view(B * N, T, -1)
        neighbor_encoded, _ = self.neighbor_encoder(neighbor_features)
        neighbor_encoded = neighbor_encoded.view(B, N, T, -1)  # [batch, max_neighbors, obs_len, hidden]
        
        # Build small-world hypergraph
        neighbor_features_last = neighbor_encoded[:, :, -1]  # Use last timestep features, shape: [batch, max_neighbors, hidden]
        connections = self.build_small_world_hypergraph(neighbor_features_last, neighbor_mask)
        
        # Perform hypergraph message passing
        neighbor_features_updated = self.hypergraph_message_passing(neighbor_features_last, connections)
        
        # Apply attention with updated features
        query = obs_encoded[:, -1].unsqueeze(1)  # [batch, 1, hidden]
        key_padding = ~neighbor_mask  # [batch, max_neighbors]
        # 对于所有邻居均被 mask 的样本，将 mask 置为全 False，避免 softmax 全部 -inf 导致 nan
        all_masked = (key_padding.sum(dim=1) == key_padding.size(1))
        if all_masked.any():
            key_padding[all_masked] = False
        attn_output, _ = self.attention(
            query,                         # [batch, 1, hidden]
            neighbor_features_updated,     # [batch, max_neighbors, hidden]
            neighbor_features_updated,     # [batch, max_neighbors, hidden]
            key_padding_mask=key_padding
        )
        
        # Combine ego and neighbor features
        combined_features = torch.cat([obs_encoded[:, -1], attn_output.squeeze(1)], dim=-1)  # [batch, hidden*2]
        
        # Process lane features using lane_encoder and fuse with combined_features
        lane_emb = self.lane_encoder(lane)            # shape: [batch, L, hidden_size]
        lane_emb_mean = lane_emb.mean(dim=1)           # aggregate lane features, shape: [batch, hidden_size]
        
        fused_features = self.context_fusion(torch.cat([combined_features, lane_emb_mean], dim=-1)) # shape: [batch, hidden*2]
        
        # Predict modes: 得到原始 logit 输出，并计算 softmax 后仅用于展示（但返回 logit 供训练使用）
        mode_logits = self.mode_predictor(fused_features)  # [batch, num_modes]
        mode_probs = F.softmax(mode_logits, dim=-1)       
        
        # Decode future trajectory for each mode
        pred_pos_all = []
        pred_vel_all = []
        
        for i in range(self.num_modes):
            # Initialize decoder with encoder states
            decoder_input = obs_encoded[:, -1:].expand(-1, self.pred_len, -1)
            decoder_output, _ = self.decoders[i](decoder_input, (h_n, c_n))
            
            # Predict positions and velocities
            pred_pos = self.pos_predictors[i](decoder_output)  # [batch, pred_len, 3]
            pred_vel = self.vel_predictors[i](decoder_output)  # [batch, pred_len, 3]
            
            pred_pos_all.append(pred_pos)
            pred_vel_all.append(pred_vel)
        
        pred_pos_all = torch.stack(pred_pos_all, dim=1)  # [batch, num_modes, pred_len, 3]
        pred_vel_all = torch.stack(pred_vel_all, dim=1)  # [batch, num_modes, pred_len, 3]
        
        # Return connections as well for small-world regularization
        return pred_pos_all, pred_vel_all, mode_logits, connections

def compute_metrics(pred_pos_all, pred_vel_all, gt_pos, gt_vel):
    """Compute evaluation metrics
    Args:
        pred_pos_all: [batch_size, num_modes, pred_len, 3]
        pred_vel_all: [batch_size, num_modes, pred_len, 3]
        gt_pos: [batch_size, pred_len, 3]
        gt_vel: [batch_size, pred_len, 3]
    Returns:
        dict: Dictionary of metrics
    """
    batch_size = pred_pos_all.size(0)
    num_modes = pred_pos_all.size(1)
    pred_len = pred_pos_all.size(2)
    
    # Time horizons for evaluation (in seconds, assuming 2Hz)
    time_horizons = [1, 2, 3, 4, 5]  # 1s, 2s, 3s, 4s, 5s
    frames_per_second = 2
    horizon_frames = [int(t * frames_per_second) for t in time_horizons]
    
    metrics = {}
    
    # Compute ADE and FDE for each mode
    ade_per_mode = []
    fde_per_mode = []
    
    for i in range(num_modes):
        # ADE (Average Displacement Error)
        displacement_error = torch.norm(pred_pos_all[:, i] - gt_pos, dim=-1)  # [batch_size, pred_len]
        ade = displacement_error.mean(dim=-1)  # [batch_size]
        ade_per_mode.append(ade)
        
        # FDE (Final Displacement Error)
        fde = displacement_error[:, -1]  # [batch_size]
        fde_per_mode.append(fde)
    
    ade_per_mode = torch.stack(ade_per_mode, dim=1)  # [batch_size, num_modes]
    fde_per_mode = torch.stack(fde_per_mode, dim=1)  # [batch_size, num_modes]
    
    # Compute minADE and minFDE
    min_ade, best_mode_ade = ade_per_mode.min(dim=1)  # [batch_size]
    min_fde, best_mode_fde = fde_per_mode.min(dim=1)  # [batch_size]
    
    # Add to metrics
    metrics["minADE"] = min_ade.mean().item()
    metrics["minFDE"] = min_fde.mean().item()
    
    # Compute velocity error metrics using best_mode_ade
    batch_size = pred_vel_all.size(0)
    selected_pred_vel = pred_vel_all[torch.arange(batch_size, device=pred_vel_all.device), best_mode_ade]  # [batch_size, pred_len, 3]
    vel_error = torch.norm(selected_pred_vel - gt_vel, dim=-1)  # [batch_size, pred_len]
    metrics["meanVelError"] = vel_error.mean().item()
    metrics["finalVelError"] = vel_error[:, -1].mean().item()
    
    # Compute RMSE for different time horizons
    for t, frames in zip(time_horizons, horizon_frames):
        if frames <= pred_len:
            # Get predictions up to current horizon
            curr_preds = pred_pos_all[:, :, :frames]  # [batch_size, num_modes, frames, 3]
            curr_gt = gt_pos[:, :frames]  # [batch_size, frames, 3]
            
            # Compute RMSE for each mode
            rmse_per_mode = []
            for i in range(num_modes):
                squared_error = ((curr_preds[:, i] - curr_gt) ** 2).sum(dim=-1)  # [batch_size, frames]
                rmse = torch.sqrt(torch.clamp(squared_error.mean(dim=-1), min=1e-6))  # [batch_size]
                rmse_per_mode.append(rmse)
            
            rmse_per_mode = torch.stack(rmse_per_mode, dim=1)  # [batch_size, num_modes]
            min_rmse, _ = rmse_per_mode.min(dim=1)  # [batch_size]
            metrics[f"RMSE_{t}s"] = min_rmse.mean().item()
    
    # Compute minADE for different k values
    k_values = [1, 5]
    for k in k_values:
        if k <= num_modes:
            # Get top-k ADEs
            topk_ade, _ = torch.topk(ade_per_mode, k=k, dim=1, largest=False)  # [batch_size, k]
            metrics[f"minADE_{k}"] = topk_ade.mean().item()
    
    return metrics

def validate(model, val_loader, device, epoch):
    """Validate the model"""
    model.eval()
    total_metrics = defaultdict(float)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating epoch {epoch}"):
            # Move data to device
            obs_pos = batch["obs_pos"].to(device)
            obs_vel = batch["obs_vel"].to(device)
            obs_acc = batch["obs_acc"].to(device)
            pred_pos = batch["pred_pos"].to(device)
            pred_vel = batch["pred_vel"].to(device)
            neighbor_pos = batch["neighbor_pos"].to(device)
            neighbor_vel = batch["neighbor_vel"].to(device)
            neighbor_mask = batch["neighbor_mask"].to(device)
            lane = batch["lane"].to(device)
            
            # Forward pass
            pred_pos_all, pred_vel_all, mode_logits, connections = model(obs_pos, obs_vel, obs_acc, neighbor_pos, neighbor_vel, neighbor_mask, lane)
            
            # Compute metrics
            metrics = compute_metrics(pred_pos_all, pred_vel_all, pred_pos, pred_vel)
            
            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] += v
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= len(val_loader)
    
    return total_metrics

def train(model, train_loader, val_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_metrics = defaultdict(float)
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        optimizer.zero_grad()
        
        # Move data to device
        obs_pos = batch["obs_pos"].to(device)
        obs_vel = batch["obs_vel"].to(device)
        obs_acc = batch["obs_acc"].to(device)
        pred_pos = batch["pred_pos"].to(device)
        pred_vel = batch["pred_vel"].to(device)
        neighbor_pos = batch["neighbor_pos"].to(device)
        neighbor_vel = batch["neighbor_vel"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)
        lane = batch["lane"].to(device)
        
        # Forward pass, now receiving connections
        pred_pos_all, pred_vel_all, mode_logits, connections = model(obs_pos, obs_vel, obs_acc, neighbor_pos, neighbor_vel, neighbor_mask, lane)
        
        # Compute trajectory prediction loss for each mode
        pos_losses = []
        vel_losses = []
        
        for i in range(model.num_modes):
            # L2 distance for positions with time-based weighting
            pos_diff = pred_pos_all[:, i] - pred_pos  # [batch, pred_len, 3]
            pos_loss = torch.norm(pos_diff, p=2, dim=-1)  # [batch, pred_len]
            
            # Progressive time weights (increasing weight for later timesteps)
            time_weights = torch.linspace(1.0, 2.0, pos_loss.size(1), device=device)
            pos_loss = (pos_loss * time_weights).mean(dim=-1)  # [batch]
            
            # L2 distance for velocities
            vel_diff = pred_vel_all[:, i] - pred_vel  # [batch, pred_len, 3]
            vel_loss = torch.norm(vel_diff, p=2, dim=-1).mean(dim=-1)  # [batch]
            
            pos_losses.append(pos_loss)
            vel_losses.append(vel_loss)
        
        pos_losses = torch.stack(pos_losses, dim=1)  # [batch, num_modes]
        vel_losses = torch.stack(vel_losses, dim=1)  # [batch, num_modes]
        
        # Combined trajectory loss with velocity weight
        traj_loss = pos_losses + 0.5 * vel_losses  # [batch, num_modes]
        
        # Get best mode for each sample
        min_losses, best_modes = traj_loss.min(dim=1)  # [batch]
        
        # Mode prediction loss (cross entropy with label smoothing)
        mode_loss = F.cross_entropy(mode_logits, best_modes, label_smoothing=0.1)
        
        # Neuromodulation regularization loss
        threshold_loss = model.threshold_neuromodulator(torch.ones(1, device=device)).mean()
        connection_loss = model.connection_neuromodulator(torch.ones(model.hidden_size, device=device)).mean()
        neuro_loss = 0.01 * (threshold_loss + connection_loss)
        
        # Small-world structure regularization using connections from forward pass
        clustering_coeff = model.compute_clustering_coefficient(connections)
        path_lengths = torch.cdist(connections, connections, p=1)
        valid_path_lengths = path_lengths[path_lengths > 0]
        if valid_path_lengths.numel() > 0:
            path_mean = valid_path_lengths.mean()
        else:
            path_mean = math.log(connections.size(1))
        sw_loss = 0.01 * (clustering_coeff.mean() - 0.5).abs() + 0.01 * (path_mean - math.log(connections.size(1)))
        
        # Total loss with all components
        loss = min_losses.mean() + 0.1 * mode_loss + neuro_loss + sw_loss
        
        # Compute metrics for monitoring
        metrics = compute_metrics(pred_pos_all, pred_vel_all, pred_pos, pred_vel)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accumulate metrics
        for k, v in metrics.items():
            total_metrics[k] += v
        
        if batch_idx % 10 == 0:
            print(f"\nBatch {batch_idx}")
            print(f"Loss: {loss.item():.4f}")
            print(f"Mode Loss: {mode_loss.item():.4f}")
            print(f"Trajectory Loss: {min_losses.mean().item():.4f}")
            print(f"Neuromodulation Loss: {neuro_loss.item():.4f}")
            print(f"Small-world Loss: {sw_loss.item():.4f}")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= len(train_loader)
    
    return total_loss / len(train_loader), total_metrics

def collate_fn(batch):
    """Custom collate function for the dataloader"""
    # Collect all keys from the first sample
    keys = batch[0].keys()
    
    # Initialize the collated batch
    collated = {}
    
    for key in keys:
        if key == "num_neighbors":
            # For scalar values, convert to tensor
            collated[key] = torch.tensor([sample[key] for sample in batch])
        else:
            # Get all tensors for this key
            tensors = [sample[key] for sample in batch]
            
            # Check if all tensors have the same shape
            shapes = [t.shape for t in tensors]
            if len(set(str(s) for s in shapes)) == 1:
                # If all shapes are the same, stack normally
                collated[key] = torch.stack(tensors, dim=0)
            else:
                # For tensors with different shapes (neighbor data)
                if key in ["neighbor_pos", "neighbor_vel"]:
                    # Find maximum dimensions
                    max_neighbors = max(t.size(0) for t in tensors)
                    seq_len = tensors[0].size(1)
                    feat_dim = tensors[0].size(2)
                    
                    # Create padded tensor
                    padded = torch.zeros(len(tensors), max_neighbors, seq_len, feat_dim)
                    mask = torch.zeros(len(tensors), max_neighbors, dtype=torch.bool)
                    
                    # Fill in the data and mask
                    for i, tensor in enumerate(tensors):
                        num_neighbors = tensor.size(0)
                        padded[i, :num_neighbors] = tensor
                        mask[i, :num_neighbors] = True
                    
                    collated[key] = padded
                    if key == "neighbor_pos":
                        collated["neighbor_mask"] = mask
                else:
                    # For other tensors that should have the same size
                    raise ValueError(f"Inconsistent tensor sizes for key {key}: {shapes}")
    
    return collated

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    full_dataset = NuScenesDataset(
        "data/processed/nuscenes-mini",
        obs_len=20,  # 4 seconds
        pred_len=30,  # 6 seconds
        distance_threshold=50.0,
        augment=True,
        max_neighbors=20  # Set maximum number of neighbors
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"\nDataset split:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    
    # Create model
    model = NEST(
        obs_len=20,
        pred_len=30,
        hidden_size=256,
        num_modes=6,
        num_heads=8
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_metrics = {
        "minADE": float('inf'),
        "minFDE": float('inf'),
        "RMSE_1s": float('inf')
    }
    num_epochs = 100
    early_stop_patience = 10
    no_improvement = 0
    
    print("\nStarting training...")
    print("=" * 50)
    print("Training parameters:")
    print(f"- Batch size: 128")
    print(f"- Initial learning rate: 0.0001")
    print(f"- Optimizer: AdamW with weight decay 0.01")
    print(f"- LR scheduler: CosineAnnealingLR")
    print(f"- Number of epochs: {num_epochs}")
    print(f"- Early stopping patience: {early_stop_patience}")
    print(f"- Observation length: 20 frames (4s)")
    print(f"- Prediction length: 30 frames (6s)")
    print(f"- Model hidden size: 256")
    print(f"- Number of modes: 6")
    print(f"- Number of attention heads: 8")
    print("=" * 50)
    
    # Training history
    history = {
        "train_loss": [],
        "val_metrics": [],
        "learning_rates": []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_metrics = train(model, train_loader, val_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device, epoch)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)
        history["learning_rates"].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print("-" * 30)
        print(f"Learning rate: {current_lr:.6f}")
        print("\nTraining metrics:")
        for k, v in train_metrics.items():
            print(f"- {k}: {v:.4f}")
        print("\nValidation metrics:")
        for k, v in val_metrics.items():
            print(f"- {k}: {v:.4f}")
        
        # Check for improvement
        improved = False
        for metric_name in best_val_metrics:
            if val_metrics[metric_name] < best_val_metrics[metric_name]:
                improved = True
                best_val_metrics[metric_name] = val_metrics[metric_name]
        
        if improved:
            no_improvement = 0
            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
                "best_val_metrics": best_val_metrics,
                "history": history
            }
            torch.save(checkpoint, checkpoint_dir / "nest_best.pt")
            print("\nSaved new best model!")
        else:
            no_improvement += 1
            if no_improvement >= early_stop_patience:
                print(f"\nEarly stopping triggered after {early_stop_patience} epochs without improvement")
                break
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
                "best_val_metrics": best_val_metrics,
                "history": history
            }
            torch.save(checkpoint, checkpoint_dir / f"nest_epoch_{epoch+1}.pt")
            
            # Save training history
            history_path = checkpoint_dir / "training_history.pkl"
            with open(history_path, "wb") as f:
                pickle.dump(history, f)
    
    print("\nTraining completed!")
    print("=" * 50)
    print("Best validation metrics:")
    for k, v in best_val_metrics.items():
        print(f"- {k}: {v:.4f}")
    
    # Save final model
    final_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_metrics": val_metrics,
        "best_val_metrics": best_val_metrics,
        "history": history
    }
    torch.save(final_checkpoint, checkpoint_dir / "nest_final.pt")

if __name__ == "__main__":
    main() 