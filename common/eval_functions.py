from benchmark_utils import iccad2015_utils, openroad_utils
from flow_matching import policies
from flow_matching.utils import check_legality_new, debug_plot_img, hpwl_fast, macro_hpwl, postprocess_placement, visualize_placement
import torch
import os
import pickle
import time

def compute_regularity(pos, sizes, min_bound=-1.0, max_bound=1.0):
    """
    Compute the weighted average regularity of macro placements.
    
    Args:
        pos (torch.Tensor): (V, 2) center positions in [-1, 1].
        sizes (torch.Tensor): (V, 2) widths and heights (normalized).
        min_bound (float): Canvas min edge (e.g., -1.0).
        max_bound (float): Canvas max edge (e.g., 1.0).
    
    Returns:
        float: Average regularity (higher = more centered/regular).
    """
    V = pos.shape[0]
    bound_width = max_bound - min_bound  # e.g., 2.0

    if pos.numel() == 0:
        return 0.0 
    
    # Compute areas
    areas = sizes[:, 0] * sizes[:, 1]  # (V',)
    total_area = areas.sum()
    
    # Compute lower/upper bounds from centers
    half_sizes = sizes / 2.0
    lower = pos - half_sizes  # (V', 2)
    upper = pos + half_sizes  # (V', 2)
    
    # Min distances to edges (clamp to >=0 for overhangs)
    dist_x_left = torch.clamp(lower[:, 0] - min_bound, min=0.0)
    dist_x_right = torch.clamp(max_bound - upper[:, 0], min=0.0)
    min_x_dist = torch.min(dist_x_left, dist_x_right)  # (V',)
    
    dist_y_bottom = torch.clamp(lower[:, 1] - min_bound, min=0.0)
    dist_y_top = torch.clamp(max_bound - upper[:, 1], min=0.0)
    min_y_dist = torch.min(dist_y_bottom, dist_y_top)  # (V',)
    
    # Per-macro regularity
    reg_per_macro = min_x_dist + min_y_dist  # (V',)
    
    # Weighted sum
    total_reg = (reg_per_macro * areas).sum()
    
    return (total_reg / total_area).item()



def from_index_to_benchmark_name(index, cfg):
    if "iccad2015"  in cfg.task:
        benchmark_list = [
            "superblue1", "superblue3", "superblue4", "superblue5", 
            "superblue7", "superblue10", "superblue16", "superblue18"
        ]
        assert 0 <= index < len(benchmark_list), f"Index {index} out of range for benchmark list."
        return benchmark_list[index]
    
    elif "ispd2005" in cfg.task:
        benchmark_list = ["adaptec1","adaptec2","adaptec3", "adaptec4", "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
        assert 0 <= index < len(benchmark_list), f"Index {index} out of range for benchmark list."
        return benchmark_list[index]
    elif "openroad" in cfg.task:
        benchmark_list = ["ariane133", "ariane136", "bp_be", "bp_fe", "bp", "swerv_wrapper"]
        assert 0 <= index < len(benchmark_list), f"Index {index} out of range for benchmark list."
        return benchmark_list[index]
    
@torch.no_grad()
def save_outputs_with_solution(
    x_in,
    cond,
    model,
    save_folder,
    output_number_offset=0,
    policy="open_loop",
    policy_kwargs = {},
    preprocess_fn=None,
    postprocess_fn=None,
    legalization_fn=None,
    cfg = None,
    ):
    """
    x_in and cond are both assumed to be on CPU
    x_in has shape (V, 2)
    preprocess_fn: x_in, cond -> x_in, cond
    postprocess_fn: sample, cond -> sample
    Returns:
    - metrics: Dict
    - sample: (V, 2) tensor 
    - cond: Data object
    All outputs are after preprocessing, and before postprocessing. tensors are on cpu
    """
    idx = cond.file_idx if "file_idx" in cond else output_number_offset
    x_in = torch.unsqueeze(x_in, dim=0).to(model.device)
    original_device = cond.x.device
    cond.to(model.device)
    metrics = {}
    metrics_special = {} # For things that should not be aggregated like plots, images, etc.

    # user-defined preprocess function
    t0 = time.time()
    x_preprocessed, cond_preprocessed = preprocess_fn(x_in, cond) if preprocess_fn is not None else (x_in, cond)

    t1 = time.time()
    if cond_preprocessed.num_nodes == 0:
        # handle edge case with 0 nodes after preprocessing
        sample = torch.zeros_like(x_preprocessed)
    else:
        if policy == "open_loop":
            sample, _, policy_metrics_special = policies.open_loop(1, model, x_preprocessed, cond_preprocessed, intermediate_every = 0, save_videos = policy_kwargs["save_videos"])
            metrics_special.update(policy_metrics_special)
        elif policy == "random":
            sample = policies.random(1, x_preprocessed, cond_preprocessed)
        else:
            raise NotImplementedError
    t2 = time.time()

    # save image too
    image = visualize_placement(sample[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))

    # legalization
    if legalization_fn is not None:
        sample, legalization_metrics, legalization_metrics_special = legalization_fn(sample, cond_preprocessed)
        metrics.update(legalization_metrics)
        metrics_special.update(legalization_metrics_special)
        image_legalized = visualize_placement(sample[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))
    else:
        image_legalized = image
    debug_plot_img(image_legalized, os.path.join(save_folder, f"placed{idx}"))

    # user-defined postprocess function
    sample_unprocessed = sample.detach().clone()
    sample, cond_postprocessed = postprocess_fn(sample, cond_preprocessed)

    sample = sample.squeeze(dim=0).detach().to(device = cond.x.device)
    sample = postprocess_placement(sample, cond_postprocessed).cpu().numpy() # mandatory post-processing
    save_file = os.path.join(save_folder, f"sample{idx}.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(sample, f)
    t3 = time.time()

    # evaluate sample and generate sampling metrics
    hpwl_normalized, hpwl_rescaled = hpwl_fast(sample_unprocessed[0], cond_preprocessed, normalized_hpwl=False)
    macro_hpwl_normalized, macro_hpwl_rescaled = macro_hpwl(sample_unprocessed[0], cond_preprocessed, normalized_hpwl=False)
    legality = check_legality_new(sample_unprocessed[0], x_in[0], cond_preprocessed, cond_preprocessed.is_ports, score=True)
    if "is_macros" in cond:
        macro_legality = check_legality_new(sample_unprocessed[0], x_in[0], cond_preprocessed, (~cond_preprocessed.is_macros) | cond_preprocessed.is_ports, score=True)
    else:
        macro_legality = 0.0
    original_hpwl_normalized = hpwl_fast(x_preprocessed, cond_preprocessed, normalized_hpwl=True)


    original_regularity = compute_regularity(x_preprocessed[0], cond_preprocessed.x)
    sample_regularity = compute_regularity(sample_unprocessed[0], cond_preprocessed.x)
    metrics.update({
        "original_regularity": original_regularity,
        "sample_regularity": sample_regularity,
    })

    t4 = time.time()
    cond.to(original_device)

    metrics.update({
        "idx": idx,
        "hpwl_normalized": hpwl_normalized,
        "hpwl_rescaled": hpwl_rescaled,
        "macro_hpwl_normalized": macro_hpwl_normalized,
        "macro_hpwl_rescaled": macro_hpwl_rescaled,
        "legality_2": legality,
        "macro_legality": macro_legality,
        "original_hpwl_normalized": original_hpwl_normalized,
        "hpwl_ratio": hpwl_normalized/max(1e-12, original_hpwl_normalized),
        "model_time": t2-t1,
        "generation_time": t3-t0,
        "eval_time": t4-t3,
        "model_vertices": cond_preprocessed.num_nodes, # number of vertices that model input has
        "model_edges": cond_preprocessed.num_edges, # number of edges that model input has
    })
    return metrics, metrics_special, image, image_legalized, sample