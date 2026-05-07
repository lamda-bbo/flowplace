import os
import sys
sys.path.append('../')
sys.path.append('../flow_matching')
import numpy as np
import torch
from torch_geometric.data import Data
from itertools import combinations
from collections import defaultdict
import pickle
from PIL import Image


# Define Args class as in the original
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Extended PlaceDBInstanceNangate45 with macro_pos generation
class PlaceDBInstanceNangate45:
    def __init__(self, args, design_name, path) -> None:
        self.args = args
        self.design_name = design_name

        benchmark_info = torch.load(path, weights_only=False)
        self.benchmark_info = benchmark_info
        self.canvas_size_x = benchmark_info["ratio_x"] * args.grid  # Assuming grid=224 as in ICCAD
        self.canvas_size_y = benchmark_info["ratio_y"] * args.grid
        self.grid_size_x = self.canvas_size_x / args.n_grid if hasattr(args, 'n_grid') else benchmark_info["ratio_x"]
        self.grid_size_y = self.canvas_size_y / args.n_grid if hasattr(args, 'n_grid') else benchmark_info["ratio_y"]

        self.node_info = benchmark_info["node_info"]
        self.net_info = benchmark_info["net_info"]
        self.node_to_net_dict = benchmark_info["node_to_net_dict"]  # Note: renamed to match ICCAD convention
        # print macro nums, net nums
        print(f"Number of macros: {len(benchmark_info['node_id_to_name'])}")
        print(f"Number of nets: {len(benchmark_info['net_info'])}")
        self.macro_names = benchmark_info["node_id_to_name"]
        self.place_order = benchmark_info["node_id_to_name"]
        self.macro_name2index_map = {name: i for i, name in enumerate(self.macro_names)}

        self.macro_size_x = []
        self.macro_size_y = []
        for macro in self.macro_names:
            self.macro_size_x.append(self.node_info[macro]["x"])
            self.macro_size_y.append(self.node_info[macro]["y"])
        self.macro_size_x = np.array(self.macro_size_x)
        self.macro_size_y = np.array(self.macro_size_y)

        self.macro_size_grid_x = np.ceil(np.maximum(1, self.macro_size_x / self.grid_size_x)).astype(np.int32)
        self.macro_size_grid_y = np.ceil(np.maximum(1, self.macro_size_y / self.grid_size_y)).astype(np.int32)
        
        # Additional attributes from benchmark_info
        self.macro_clusters = benchmark_info["macro_clusters"]
        self.dataflow_mat = benchmark_info["dataflow_mat"]
        self.id2index = benchmark_info["id2index"]
        self.port_pos = benchmark_info["port_pos"]
        self.pin_blocking_rectangles = benchmark_info["pin_blocking_rectangles"]
        

        self.ratio_x = self.canvas_size_x / args.grid
        self.ratio_y = self.canvas_size_y / args.grid
        self.ratio_sum = self.ratio_x + self.ratio_y

        # Build PyG data (similar to ICCAD's _build_pyg_data)
        self.x, self.cond = self._build_pyg_data()

    def _build_pyg_data(self):
        V = len(self.macro_names)
        node_name_to_id = {name: i for i, name in enumerate(self.macro_names)}

        # Normalize sizes to [0, 2] corresponding to chip normalized to [-1, 1]
        # Using 'x' and 'y' from node_info as sizes (similar to ICCAD)
        sizes = []
        max_width = self.canvas_size_x
        max_height = self.canvas_size_y
        for name in self.macro_names:
            sx = 2 * self.node_info[name]['x'] / max_width
            sy = 2 * self.node_info[name]['y'] / max_height
            sizes.append([sx, sy])
        cond_x = torch.tensor(sizes, dtype=torch.float32)

        # is_ports all 0 (assuming no ports as in ICCAD)
        is_ports = torch.zeros((V, 1), dtype=torch.float32)

        # Normalize initial positions (centers) to [-1, 1]
        # Using 'raw_x' and 'raw_y' as initial positions, add size/2 for center
        positions = []
        for name in self.macro_names:
            center_x = self.node_info[name].get("raw_x", 0) + self.node_info[name]['x'] / 2
            center_y = self.node_info[name].get("raw_y", 0) + self.node_info[name]['y'] / 2
            norm_x = 2 * center_x / max_width - 1
            norm_y = 2 * center_y / max_height - 1
            positions.append([norm_x, norm_y])
        x = torch.tensor(positions, dtype=torch.float32)

        # Build edges and attrs using net_info directly (similar to ICCAD)
        edge_list = []
        attr_list = []
        for net_name in self.net_info:
            macro_to_off = {}
            for macro in self.net_info[net_name]["nodes"]:
                off_x = self.net_info[net_name]["nodes"][macro].get("x_offset", 0)
                off_y = self.net_info[net_name]["nodes"][macro].get("y_offset", 0)
                norm_off_x = 2 * off_x / max_width
                norm_off_y = 2 * off_y / max_height
                macro_to_off[macro] = (norm_off_x, norm_off_y)

            connected_macros = list(macro_to_off.keys())
            if len(connected_macros) < 2:
                continue

            for a, b in combinations(connected_macros, 2):
                off_a = macro_to_off[a]
                off_b = macro_to_off[b]
                i = node_name_to_id[a]
                j = node_name_to_id[b]
                edge_list.append([i, j])
                attr_list.append([off_a[0], off_a[1], off_b[0], off_b[1]])
                edge_list.append([j, i])
                attr_list.append([off_b[0], off_b[1], off_a[0], off_a[1]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(attr_list, dtype=torch.float32)
        else:
            print("Warning: no edges in the graph.")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)

        cond = Data(x=cond_x, edge_index=edge_index, edge_attr=edge_attr, is_ports=is_ports)
        chip_size = np.array([0.0, 0.0, 2.0, 2.0])  # normalized to [-1, 1]
        cond.chip_size = chip_size
        cond.benchmark_name = self.design_name  # Set benchmark_name
        return x, cond

    def get_macro_pos(self, in_grid_space=False):
        """
        Generate macro_pos dict.
        - If in_grid_space=True, use prototype grid positions from 'macro_pos_prototype' (e.g., (x, y, w, h) in grid).
        - If in_grid_space=False, use original space positions: raw_x, raw_y, x (size), y (size).
        """
        macro_pos = {}
        if in_grid_space:
            # Use macro_pos_prototype if available, assuming it's in grid space
            if "macro_pos_prototype" in self.benchmark_info:
                macro_pos = self.benchmark_info["macro_pos_prototype"]
            else:
                # Fallback: compute grid positions from raw_x/y and sizes
                for name in self.macro_names:
                    grid_x = int(self.node_info[name].get("raw_x", 0) / self.grid_size_x)
                    grid_y = int(self.node_info[name].get("raw_y", 0) / self.grid_size_y)
                    grid_w = self.macro_size_grid_x[self.macro_name2index_map[name]]
                    grid_h = self.macro_size_grid_y[self.macro_name2index_map[name]]
                    macro_pos[name] = (grid_x, grid_y, grid_w, grid_h)
        else:
            # Original space: (raw_x, raw_y, size_x, size_y)
            for name in self.macro_names:
                orig_x = self.node_info[name].get("raw_x", 0)
                orig_y = self.node_info[name].get("raw_y", 0)
                size_x = self.node_info[name]['x']
                size_y = self.node_info[name]['y']
                macro_pos[name] = (orig_x, orig_y, size_x, size_y)
        return macro_pos
    
def convert_to_openroad(sample, benchmark_name, output_dir):
    """
    Convert a placement `sample` (N×2, normalized in [-1, 1]) to OpenROAD format
    and save it under `output_dir` as `<benchmark_name>_placement.pt`.

    The saved dict contains:
        macro_pos : {macro_id: (llx, lly, width, height)}   # all in *original* physical units
    """
    # ------------------------------------------------------------------ #
    # 1. 重新加载对应的 benchmark（只需要 canvas / ratio / macro info）
    # ------------------------------------------------------------------ #

    args = Args(BENCHMARK_DIR="benchmark_cache", grid=224, n_grid=224)
    pt_path = os.path.join(args.BENCHMARK_DIR, f"{benchmark_name}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Benchmark file not found: {pt_path}")

    benchmark_info = torch.load(pt_path, weights_only=False)

    canvas_w = benchmark_info["ratio_x"] * args.grid   # = canvas_size_x
    canvas_h = benchmark_info["ratio_y"] * args.grid   # = canvas_size_y

    macro_names = benchmark_info["node_id_to_name"]
    node_info   = benchmark_info["node_info"]

    # ------------------------------------------------------------------ #
    # 2. 把 normalized center → physical lower-left corner
    # ------------------------------------------------------------------ #
    # sample: [N, 2]  (norm_x, norm_y)  ∈ [-1, 1]
    # → physical center = (norm + 1) * canvas / 2
    if isinstance(sample, np.ndarray):
        sample = torch.from_numpy(sample)
    centers_phys = (sample + 1.0) * torch.stack([torch.full_like(sample[:, 0], canvas_w),
                                                torch.full_like(sample[:, 1], canvas_h)], dim=1) / 2.0

    macro_pos = {}
    for idx, macro_id in enumerate(macro_names):
        cx, cy = centers_phys[idx]                     # physical center
        w = node_info[macro_id]["x"]                   # width  (original unit)
        h = node_info[macro_id]["y"]                   # height (original unit)

        llx = (cx - w / 2.0 - benchmark_info["ratio_x"]) / benchmark_info["ratio_x"]
        lly = (cy - h / 2.0 -  benchmark_info["ratio_y"]) / benchmark_info["ratio_y"]
        macro_pos[macro_id] = (float(llx), float(lly), float(w/benchmark_info["ratio_x"])  , float(h/benchmark_info["ratio_y"]))

    # ------------------------------------------------------------------ #
    # 3. 保存
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{benchmark_name}.pt")
    torch.save(macro_pos, save_path)
    print(f"OpenROAD placement saved to: {save_path}")
    return macro_pos


if __name__ == "__main__":
    from flow_matching import utils

# Assuming the following utilities are available from the original codebase
# If not, they need to be implemented or imported accordingly
    from flow_matching.utils import visualize_placement, hpwl_fast, hpwl  # Assuming these exist
    # Main script to process benchmarks, similar to ICCAD2015
    args = Args(BENCHMARK_DIR="../benchmark_cache",  # Adjust path as needed
                grid=224,
                n_grid=224) 

    
    output_dir = "../datasets/graph/openroad-s0"  # s for self-converted
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    problem_instances = {}
    benchmark_list = ["ariane133", "ariane136", "bp_be", "bp_fe", "bp", "swerv_wrapper"]
    for idx, benchmark_name in enumerate(benchmark_list):
        print(f"\n{'='*50}")
        print(f"Processing {benchmark_name}")
        print(f"{'='*50}")
        
        # Assume paths to .pt files, e.g., f"{args.BENCHMARK_DIR}/{benchmark_name}.pt"
        path = os.path.join(args.BENCHMARK_DIR, f"{benchmark_name}.pt")  # Adjust if different
        
        # Create problem instance using the extended class
        problem_instance = PlaceDBInstanceNangate45(args, benchmark_name, path)
        problem_instances[benchmark_name] = problem_instance

        coords, cond = problem_instance.x, problem_instance.cond
        cond.benchmark_name = benchmark_name  # Already set, but ensure
        cond_path = os.path.join(output_dir, f"graph{idx}.pickle")
        x_path = os.path.join(output_dir, f"output{idx}.pickle")

        with open(cond_path, 'wb') as f:
            pickle.dump(cond, f)
        with open(x_path, 'wb') as f:
            pickle.dump(coords, f)
        print(f"Successfully saved graph and coordinates for {benchmark_name}.")

        print(f"max, min of coords: {coords.max()}, {coords.min()}")
        print(f"max, min of cond.x: {cond.x.max()}, {cond.x.min()}")
        
        # Visualize (assuming visualize_placement exists)
        img = visualize_placement(coords, cond, True, True)
        Image.fromarray(img).show()
        img_without_net = visualize_placement(coords, cond, False, False)
        Image.fromarray(img_without_net).show()


        # Example: Get macro_pos in both spaces
        macro_pos_original = problem_instance.get_macro_pos(in_grid_space=False)
        macro_pos_grid = problem_instance.get_macro_pos(in_grid_space=True)
        print(f"Sample macro_pos (original): {list(macro_pos_original.items())[:1]}")
        print(f"Sample macro_pos (grid): {list(macro_pos_grid.items())[:1]}")

