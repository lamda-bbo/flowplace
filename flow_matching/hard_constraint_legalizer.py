import torch
import torch.nn.functional as F

@torch.jit.script
def minmax_norm(vec: torch.Tensor) -> torch.Tensor:
    vmin = vec.min()
    vmax = vec.max()
    if vmax - vmin < 1e-8:
        return torch.zeros_like(vec)
    return (vec - vmin) / (vmax - vmin)

@torch.jit.script
def compute_overlap_mask(candidate_bounds: torch.Tensor, placed_bounds: torch.Tensor) -> torch.Tensor:
    N_cand, N_placed = candidate_bounds.shape[0], placed_bounds.shape[0]
    
    #  (N_cand, N_placed, 4)
    cand_exp = candidate_bounds.unsqueeze(1).expand(-1, N_placed, -1)
    placed_exp = placed_bounds.unsqueeze(0).expand(N_cand, -1, -1)
    
    overlap_x = (cand_exp[..., 0] < placed_exp[..., 2]) & (cand_exp[..., 2] > placed_exp[..., 0])
    overlap_y = (cand_exp[..., 1] < placed_exp[..., 3]) & (cand_exp[..., 3] > placed_exp[..., 1])
    
    return (overlap_x & overlap_y).any(dim=1)  # (N_cand,)

def candidate_distance_score(orig_pos, candidates):
    """
    orig_pos: (2,) original model output position (center)
    candidates: (K,2)
    returns: dist_normed: (K,)  
    """
    diffs = candidates - orig_pos.unsqueeze(0)  # (K,2)
    d2 = (diffs ** 2).sum(dim=1)               # (K,)
    normalized = d2 / 8.0
    return normalized

def regularity_cost_for_macro(candidates, sizes):
    """
    candidates: (K,2)
    sizes: (2,) w,h
    returns: reg_cost: (K,) 
    """
    w, h = float(sizes[0]), float(sizes[1])
    corners = torch.tensor([[-1 + w/2, -1 + h/2],
                            [-1 + w/2,  1 - h/2],
                            [ 1 - w/2, -1 + h/2],
                            [ 1 - w/2,  1 - h/2]], device=candidates.device, dtype=candidates.dtype)  # (4,2)
    d_corner = torch.abs(candidates.unsqueeze(1) - corners.unsqueeze(0)).sum(dim=2)  # (K,4)
    min_corner_dist = d_corner.min(dim=1).values  # (K,)
    reg_cost = min_corner_dist / 4.0
    return reg_cost

class Legalizer:
    def __init__(self, cond, grid_res=32, scores=None, device=None, dtype=torch.float32, padding=0.005):
        self.scores = {
            "w_legality": 1e6,
            "w_hpwl": 1.0,
            "w_reg": 0.0,
            "w_dist": 1.0,
        } if scores is None else scores
        self.device = device if device is not None else cond.x.device
        self.dtype = dtype
        self.padding = padding
        
        self.sizes = cond.x.to(self.device, self.dtype)  # (V,2)
        self.fixed_mask = cond.is_ports.squeeze(-1).bool().to(self.device)

        axis = torch.linspace(-1, 1, grid_res, device=self.device, dtype=self.dtype)
        X, Y = torch.meshgrid(axis, axis, indexing='ij')
        self.grid = torch.stack([X, Y], dim=-1).reshape(-1, 2)  # (G,2)
    
        self.G = self.grid.shape[0]

        V = self.sizes.shape[0]
        self.placed_bounds = torch.empty((V, 4), device=self.device, dtype=self.dtype)

        areas = (self.sizes[:, 0] * self.sizes[:, 1])
        movable = (~self.fixed_mask).nonzero(as_tuple=True)[0]
        movable_sorted = movable[areas[movable].argsort(descending=True)]
        fixed = (self.fixed_mask).nonzero(as_tuple=True)[0]
        self.order = torch.cat([fixed, movable])

        w = self.sizes[:, 0]
        h = self.sizes[:, 1]

        corners_all = torch.stack([
            torch.stack([-1 + w/2, -1 + h/2], dim=1),  
            torch.stack([-1 + w/2,  1 - h/2], dim=1),  
            torch.stack([ 1 - w/2, -1 + h/2], dim=1),  
            torch.stack([ 1 - w/2,  1 - h/2], dim=1),  
        ], dim=1)  

        grid_exp = self.grid.unsqueeze(0).unsqueeze(2)  # (1,G,1,2)
        corners_exp = corners_all.unsqueeze(1)  # (V,1,4,2)

        d_corner = torch.abs(grid_exp - corners_exp).sum(dim=3)  # (V,G,4)
        min_corner_dist = d_corner.min(dim=2).values  # (V,G)
        self.reg_cost_cache = min_corner_dist / 4.0

    def run(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        x = x.clone()
        
        placed_count = 0

        with torch.no_grad(): 
            for idx in self.order:
                w, h = self.sizes[idx]
                orig_pos = x[0, idx]

                w_eff = w + 2 * self.padding
                h_eff = h + 2 * self.padding

                if self.fixed_mask[idx]:
                    new_pos = orig_pos
                else:
                    diff = self.grid - orig_pos
                    d2_all = (diff**2).sum(dim=1) / 8.0 

                    base_order = d2_all.argsort()
                    candidates = self.grid[base_order]
                    d2_all = d2_all[base_order]
                    reg_all = self.reg_cost_cache[idx][base_order]

                    valid = (
                        (candidates[:,0] >= -1 + w_eff/2) &
                        (candidates[:,0] <=  1 - w_eff/2) &
                        (candidates[:,1] >= -1 + h_eff/2) &
                        (candidates[:,1] <=  1 - h_eff/2)
                    )

                    if placed_count > 0:
                        cand_bounds = torch.stack([
                            candidates[:,0] - w_eff/2, 
                            candidates[:,1] - h_eff/2,
                            candidates[:,0] + w_eff/2, 
                            candidates[:,1] + h_eff/2
                        ], dim=-1)
                        overlap = compute_overlap_mask(cand_bounds, self.placed_bounds[:placed_count])
                    else:
                        overlap = torch.zeros_like(valid)

                    mask = valid & (~overlap)

                    if not mask.any():
                        new_pos = orig_pos
                    else:
                        cand_final = candidates[mask]
                        dist_cost = d2_all[mask]
                        reg_cost = reg_all[mask]

                        dist_cost = minmax_norm(dist_cost)
                        reg_cost = minmax_norm(reg_cost)

                        score = -(self.scores['w_dist'] * dist_cost + self.scores['w_reg'] * reg_cost)
                        new_pos = cand_final[score.argmax()]

                x[0, idx] = new_pos
                self.placed_bounds[placed_count] = torch.tensor(
                    [new_pos[0] - w_eff/2, new_pos[1] - h_eff/2, new_pos[0] + w_eff/2, new_pos[1] + h_eff/2],
                    device=self.device, dtype=self.dtype
                )
                placed_count += 1

        return x
