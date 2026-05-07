from distributions import get_distribution
import utils
import torch
import shapely
import numpy as np
from torch_geometric.data import Data
import torch_geometric.utils as tgu
import math

class MaskGuidedDataGeneration:
    def __init__(
            self, 
            max_instance,
            stop_density_dist,
            max_attempts_per_instance,
            aspect_ratio_dist, 
            instance_size_dist, 
            num_terminals_dist,
            edge_dist,
            source_terminal_dist,
            interior_terminals_dist=None,
            interior_terminals_loc="uniform",
            zero_edge_attr=False,
            distance_norm_order=1,
            grid_size=128,
            weight_mode="dist_to_boundary",
    ):
        self.max_instance = max_instance
        self.stop_density_dist = stop_density_dist
        self.aspect_ratio_dist = aspect_ratio_dist
        self.instance_size_dist = instance_size_dist
        self.num_terminals_dist = num_terminals_dist
        self.interior_terminals_dist = interior_terminals_dist
        self.interior_terminals_loc = interior_terminals_loc
        self.edge_dist = edge_dist
        self.max_attempts_per_instance = max_attempts_per_instance
        self.source_terminal_dist = source_terminal_dist
        self.zero_edge_attr = zero_edge_attr
        self.distance_norm_order = distance_norm_order
        self.grid_size = grid_size
        self.weight_mode = weight_mode
        # Precompute weight matrix at initialization
        self.weight_matrix = self._compute_weight_matrix(grid_size)
    
    def get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals, reference="center"):
        half_perim = (x_sizes + y_sizes)
        terminal_locations = get_distribution("uniform", {"low": 0, "high": half_perim}).sample((max_num_terminals,))
        terminal_flip = get_distribution("bernoulli", {"probs": 0.5}).sample((max_num_terminals, x_sizes.shape[0]))
        terminal_flip = (2 * terminal_flip) - 1

        x_sizes = x_sizes.unsqueeze(dim=0)
        y_sizes = y_sizes.unsqueeze(dim=0)
        boundary_offset_x = torch.clamp(terminal_locations, torch.zeros_like(x_sizes), x_sizes) - (x_sizes/2) 
        boundary_offset_y = torch.clamp(terminal_locations-x_sizes, torch.zeros_like(y_sizes), y_sizes) - (y_sizes/2)

        boundary_offset_x = terminal_flip * boundary_offset_x
        boundary_offset_y = terminal_flip * boundary_offset_y

        boundary_offset = torch.stack((boundary_offset_x, boundary_offset_y), dim=-1).movedim(1, 0)
        
        if self.interior_terminals_dist is not None:
            sizes = torch.stack((x_sizes, y_sizes), dim=-1).squeeze(dim=0)
            gm_size = torch.sqrt(x_sizes * y_sizes).squeeze(dim=0)
            is_terminal_interior = get_distribution(**self.interior_terminals_dist).sample(gm_size).view(gm_size.shape[0], 1, 1)
            if self.interior_terminals_loc == "uniform":
                interior_offset = get_distribution("uniform", {"low": -sizes/2, "high": sizes/2}).sample((max_num_terminals,))
                interior_offset = interior_offset.moveaxis(0, 1)
            elif self.interior_terminals_loc == "center":
                interior_offset = torch.zeros_like(boundary_offset)
            else:
                raise NotImplementedError
            terminal_offset = is_terminal_interior * interior_offset + (1-is_terminal_interior) * boundary_offset
        else:
            terminal_offset = boundary_offset
        
        if reference == "bottom_left":
            terminal_offset[:,:,0] += x_sizes/2
            terminal_offset[:,:,1] += y_sizes/2
        return terminal_offset

    def get_terminal_distances(self, terminal_positions, norm_order=1):
        if norm_order == "inf":
            norm_order = float(norm_order)
        V, T, _ = terminal_positions.shape
        t_pos_1 = terminal_positions.view(V, T, 1, 1, 2)
        t_pos_2 = terminal_positions.view(1, 1, V, T, 2)
        delta_pos = t_pos_1 - t_pos_2
        distance = torch.norm(delta_pos, p=norm_order, dim=-1)
        return distance

    def process_edge_matrix(self, edge_exists, is_source, num_terminals):
        V, T, _, _ = edge_exists.shape
        assert is_source.shape == edge_exists.shape[:2]
        assert num_terminals.shape == (V,)

        terminal_filter = torch.zeros((V, T))
        for i, num_terminal in enumerate(num_terminals):
            terminal_filter[i, :num_terminal] = 1

        source_filter = (terminal_filter * is_source).view(V, T, 1, 1)
        sink_filter = (terminal_filter * (1-is_source)).view(1, 1, V, T)
        self_edge_filter = (1-torch.eye(V)).view(V, 1, V, 1)
        
        edges = edge_exists * source_filter
        edges = edges * sink_filter
        edges = edges * self_edge_filter
        return edges

    def connect_isolated_instances(self, edge_matrix, terminal_distances):
        V, T, _, _ = edge_matrix.shape
        out_degree = edge_matrix.sum(dim=(2,3))
        in_degree = edge_matrix.sum(dim=(0,1,3))
        degree = out_degree.sum(dim=-1) + in_degree
        max_dist = 10 + terminal_distances.max()
        for i in range(V):
            if degree[i] == 0:
                distances = terminal_distances[i, 0, :, :]
                distances = torch.where(out_degree > 0, distances, max_dist)
                
                min_idx = torch.argmin(distances)
                instance_idx = min_idx // T
                terminal_idx = min_idx % T
                edge_matrix[instance_idx, terminal_idx, i, 0] = 1

    def generate_edge_list(self, edge_exists, terminal_offsets):
        V, T, _, _ = edge_exists.shape
        edges = torch.nonzero(edge_exists)
        edge_index_forward = edges[:,(0,2)]
        edge_index_reverse = edges[:,(2,0)]

        edge_attr_source = terminal_offsets[edges[:,0], edges[:,1], :]
        edge_attr_sink = terminal_offsets[edges[:,2], edges[:,3], :]
        edge_attr_forward = torch.concat((edge_attr_source, edge_attr_sink), dim=-1)
        edge_attr_reverse = torch.concat((edge_attr_sink, edge_attr_source), dim=-1)
        
        edge_index = torch.concat((edge_index_forward, edge_index_reverse), dim=0).T
        edge_attr = torch.concat((edge_attr_forward, edge_attr_reverse), dim=0)
        
        return edge_index.clone(), edge_attr.clone()
    
    def sample(
            self, 
            size_dist_timer=None,
            place_timer=None,
            terminal_timer=None,
            edge_timer=None,
    ):
        #print("Sampling new layout...")
        size_dist_timer.start() if size_dist_timer else None

        # Generate stop density
        stop_density = get_distribution(**self.stop_density_dist).sample()

        # Generate instance sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,))
        long_size = get_distribution(**self.instance_size_dist).sample((self.max_instance,))
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((self.max_instance,))

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

        # Sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        size_dist_timer.stop() if size_dist_timer else None
        place_timer.start() if place_timer else None

        # Initialize grid-based placement
        placement = GridPlacement(self.grid_size)
        density = 0
        
        # Place instances using grid-based approach
        for i, (x_size, y_size) in enumerate(zip(x_sizes, y_sizes)):
            x_size = float(x_size)
            y_size = float(y_size)

            # Convert to grid units (half canvas in [-1,1])
            x_size_grid = max(1, math.ceil(x_size * self.grid_size / 2.0))
            y_size_grid = max(1, math.ceil(y_size * self.grid_size / 2.0))
            
            position_mask = placement.get_valid_position_mask(x_size_grid, y_size_grid, x_size, y_size)
            valid_indices = torch.nonzero(position_mask.flatten(), as_tuple=True)[0]

            flat_weight = self.weight_matrix.flatten()
            mask_flat = position_mask.flatten()
            # Only consider valid indices where mask == 1
            valid_indices = torch.nonzero(mask_flat, as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue
                
            valid_weights = flat_weight[valid_indices]
            if valid_weights.sum() == 0:
                continue

            valid_probs = valid_weights# / valid_weights.sum()
            sampled_idx = torch.multinomial(valid_probs, 1).item()
            flat_idx = valid_indices[sampled_idx].item()
            y_idx = flat_idx // self.grid_size
            x_idx = flat_idx % self.grid_size

            # Double check position is valid (should always be true)
            if not position_mask[y_idx, x_idx]:
                continue

            # Commit to placement
            placement.commit_instance(x_idx, y_idx, x_size, y_size, x_size_grid, y_size_grid)

            density += (x_size * y_size) / 4.0
            if density >= stop_density:
                break

        positions = placement.get_positions()
        sizes = placement.get_sizes()
        num_instances = positions.shape[0]

        place_timer.stop() if place_timer else None
        terminal_timer.start() if terminal_timer else None

        # Sample number of terminals
        instance_area = sizes[:, 0] * sizes[:, 1]
        num_terminals = get_distribution(**self.num_terminals_dist).sample(instance_area).int()
        num_terminals = torch.clip(num_terminals, min=1, max=256)
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(sizes[:,0], sizes[:,1], max_num_terminals, reference="center")

        terminal_timer.stop() if terminal_timer else None
        edge_timer.start() if edge_timer else None

        # Generate edges
        terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets
        terminal_distances = self.get_terminal_distances(terminal_positions, norm_order=self.distance_norm_order)
        edge_exists = get_distribution(**self.edge_dist).sample(terminal_distances)
        is_source = get_distribution(**self.source_terminal_dist).sample((num_instances, max_num_terminals))

        edge_exists = self.process_edge_matrix(edge_exists, is_source, num_terminals)
        self.connect_isolated_instances(edge_exists, terminal_distances)

        edge_index, edge_attr = self.generate_edge_list(edge_exists, terminal_offsets)
        mask = placement.get_mask()
        if self.zero_edge_attr:
            edge_attr = 0 * edge_attr

        edge_timer.stop() if edge_timer else None

        data = Data(x=sizes, edge_index=edge_index, edge_attr=edge_attr, is_ports=mask)
        return positions, data



    def _compute_weight_matrix(self, grid_size):
        if self.weight_mode == "regular_mask":
            # compute distance to nearest corner
            y_idx = torch.arange(grid_size).view(-1, 1).repeat(1, grid_size)
            x_idx = torch.arange(grid_size).view(1, -1).repeat(grid_size, 1)
            corners = [
                (0, 0),
                (0, grid_size - 1),
                (grid_size - 1, 0),
                (grid_size - 1, grid_size - 1)
            ]
            dists = [torch.abs(x_idx - cx) + torch.abs(y_idx - cy) for (cx, cy) in corners]
            min_dist = torch.stack(dists, dim=0).min(dim=0)[0]
            
            weights = grid_size / (min_dist + 1.0) ** 2  # avoid division by zero, and ensure corners have highest weight of grid_size

            return weights
        elif self.weight_mode == "dist_to_center":
            # compute distance to center
            center_x = (grid_size - 1) / 2.0
            center_y = (grid_size - 1) / 2.0
            y_idx = torch.arange(grid_size).view(-1, 1).repeat(1, grid_size)
            x_idx = torch.arange(grid_size).view(1, -1).repeat(grid_size, 1)
            dists = torch.abs(x_idx - center_x) + torch.abs(y_idx - center_y)
            weights = dists 

            return weights
        # compute distance to nearest boundary
        elif  self.weight_mode == "dist_to_boundary":
            y_idx = torch.arange(grid_size).view(-1, 1).repeat(1, grid_size)
            x_idx = torch.arange(grid_size).view(1, -1).repeat(grid_size, 1)
            # compute distance to four boundaries
            dist_to_left = x_idx
            dist_to_right = grid_size - 1 - x_idx
            dist_to_top = y_idx
            dist_to_bottom = grid_size - 1 - y_idx
            
            # take minimum distance as weight
            weights = grid_size / (torch.min(torch.min(dist_to_left, dist_to_right), torch.min(dist_to_top, dist_to_bottom)) +1.0) **2  # avoid division by zero, and ensure boundary positions have highest weight of grid_size
            return weights
        else:

            raise NotImplementedError(f"Weight mode {self.weight_mode} not supported")
        
class GridPlacement:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = torch.zeros((grid_size, grid_size), dtype=torch.bool)
        self.positions = []
        self.sizes = []
        self.grid_sizes = []

    def get_valid_position_mask(self, x_size_grid, y_size_grid, x_size, y_size):
        # Compute valid boundaries in grid units based on continuous size
        max_x = max(0, self.grid_size - x_size_grid)
        max_y = max(0, self.grid_size - y_size_grid)
        if max_x <= 0 or max_y <= 0:
            return torch.zeros((self.grid_size, self.grid_size), dtype=torch.bool)
        
        # Initialize mask
        mask = torch.zeros((self.grid_size, self.grid_size), dtype=torch.bool)
        
        # Use convolution to check for valid regions (left-lower corner positions)
        padded_grid = torch.nn.functional.pad(
            self.grid, (0, x_size_grid-1, 0, y_size_grid-1), mode='constant', value=1
        )
        kernel = torch.ones((y_size_grid, x_size_grid), dtype=torch.float)
        conv_result = torch.nn.functional.conv2d(
            padded_grid.unsqueeze(0).unsqueeze(0).float(),
            kernel.unsqueeze(0).unsqueeze(0),
            stride=1
        ).squeeze()
        
        # Valid positions where no overlap occurs, within boundaries
        mask[:max_y, :max_x] = (conv_result[:max_y, :max_x] == 0)
        return mask

    def commit_instance(self, x_idx, y_idx, x_size, y_size, x_size_grid, y_size_grid):
        # Clamp grid indices to ensure valid range
        x_idx = max(0, min(x_idx, self.grid_size - x_size_grid))
        y_idx = max(0, min(y_idx, self.grid_size - y_size_grid))
        
        # Convert grid left-lower corner to continuous center coordinates
        x_pos = 2 * (x_idx + x_size_grid / 2.0) / self.grid_size - 1
        y_pos = 1 - 2 * (y_idx + y_size_grid / 2.0) / self.grid_size
        
        # Mark grid cells as occupied
        self.grid[y_idx:y_idx+y_size_grid, x_idx:x_idx+x_size_grid] = True
        self.positions.append([x_pos, y_pos])
        self.sizes.append([x_size, y_size])
        self.grid_sizes.append([x_size_grid, y_size_grid])
        
        #print(f"✅ Commit @ grid ({x_idx},{y_idx}) | pos=({x_pos:.3f},{y_pos:.3f}) | size=({x_size:.3f},{y_size:.3f})")

    def get_positions(self):
        #
        return torch.tensor(self.positions, dtype=torch.float) if self.positions else torch.empty((0, 2))

    def get_sizes(self):
        return torch.tensor(self.sizes, dtype=torch.float) if self.sizes else torch.empty((0, 2))

    def get_mask(self):
        return torch.zeros((len(self.positions),), dtype=torch.bool)


import torch

class GridPlacement:
    def __init__(self, grid_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.grid_size = grid_size
        self.grid = torch.zeros((grid_size, grid_size), dtype=torch.bool)
        self.positions = []
        self.sizes = []
        self.grid_sizes = []
        self.device = device  # Store device for GPU/CPU operations

    def get_valid_position_mask(self, x_size_grid, y_size_grid, x_size, y_size):
        # Compute valid boundaries in grid units based on continuous size
        max_x = max(0, self.grid_size - x_size_grid)
        max_y = max(0, self.grid_size - y_size_grid)
        if max_x <= 0 or max_y <= 0:
            #print("⚠️ Instance too large for grid.")
            return torch.zeros((self.grid_size, self.grid_size), dtype=torch.bool)
        
        # Initialize mask
        mask = torch.zeros((self.grid_size, self.grid_size), dtype=torch.bool)
        
        # Move tensors to GPU for conv2d
        grid = self.grid.float().to(self.device)
        padded_grid = torch.nn.functional.pad(
            grid, (0, x_size_grid-1, 0, y_size_grid-1), mode='constant', value=1
        )
        kernel = torch.ones((y_size_grid, x_size_grid), dtype=torch.float, device=self.device)
        
        # Perform conv2d on GPU
        conv_result = torch.nn.functional.conv2d(
            padded_grid.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            stride=1
        ).squeeze()
        
        # Valid positions where no overlap occurs, within boundaries
        mask[:max_y, :max_x] = (conv_result[:max_y, :max_x] == 0)
        return mask

    def commit_instance(self, x_idx, y_idx, x_size, y_size, x_size_grid, y_size_grid):
        # Clamp grid indices to ensure valid range
        x_idx = max(0, min(x_idx, self.grid_size - x_size_grid))
        y_idx = max(0, min(y_idx, self.grid_size - y_size_grid))
        
        # Convert grid left-lower corner to continuous center coordinates
        x_pos = 2 * (x_idx + x_size_grid / 2.0) / self.grid_size - 1
        y_pos = 1 - 2 * (y_idx + y_size_grid / 2.0) / self.grid_size
        
        # Mark grid cells as occupied
        self.grid[y_idx:y_idx+y_size_grid, x_idx:x_idx+x_size_grid] = True
        self.positions.append([x_pos, y_pos])
        self.sizes.append([x_size, y_size])
        self.grid_sizes.append([x_size_grid, y_size_grid])
        
        #print(f"✅ Commit @ grid ({x_idx},{y_idx}) | pos=({x_pos:.3f},{y_pos:.3f}) | size=({x_size:.3f},{y_size:.3f})")

    def get_positions(self):
        return torch.tensor(self.positions, dtype=torch.float) if self.positions else torch.empty((0, 2))

    def get_sizes(self):
        return torch.tensor(self.sizes, dtype=torch.float) if self.sizes else torch.empty((0, 2))

    def get_mask(self):
        return torch.zeros((len(self.positions),), dtype=torch.bool)