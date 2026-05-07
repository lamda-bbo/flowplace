import utils
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import common
import os
import time
import legalization
import analysis_utils
import wandb

import warnings
# filter out warnings Functor warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def cost(output_metrics):
    """
    Returns dict with cost function(s) for hyperparam sweep
    """
    legality_target = 0.995
    macro_legality_target = 0.998
    legality_temp = 0.001
    hpwl = torch.tensor(output_metrics["hpwl_rescaled"]).mean()
    
    legality = torch.tensor(output_metrics["legality_2"]).mean()
    legality_cost_factor = 1 + 10 * torch.nn.functional.relu((legality_target - legality)/legality_temp)
    
    macro_legality = torch.tensor(output_metrics["macro_legality"]).mean()
    macro_legality_cost_factor = 1 + 10 * torch.nn.functional.relu((macro_legality_target - macro_legality)/legality_temp)
    
    full_cost = (legality_cost_factor * hpwl).item()
    macro_cost = (macro_legality_cost_factor * hpwl).item()
    costs = {
        "cost": full_cost,
        "macro_cost": macro_cost,
    }
    return costs

@hydra.main(version_base=None, config_path="configs", config_name="config_combined")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(sample_dir)
    except FileExistsError:
        pass
    print(f"saving checkpoints to: {log_dir}")
    torch.manual_seed(cfg.seed)

    # Prepare legalization function
    if cfg.legalization.mode in [None, "none", "None", ""]:
        legalize_fn = None
    elif cfg.legalization.mode == "scheduled":
        def legalize_fn(x, cond):
            return legalization.legalize(
                x, 
                cond,
                **cfg.legalization,
                )
    elif cfg.legalization.mode == "opt":
        def legalize_fn(x, cond):
            return legalization.legalize_opt(
                x, 
                cond,
                **cfg.legalization,
                )
    print(f"Using legalization: {cfg.legalization.mode}")
    print(f"Using guidance: {cfg.model.guidance_mode}")
    # Prepare pre and post processing functions. Note that postprocess fns are applied in reverse order
    preprocess_fns = []
    postprocess_fns = []
    if cfg.cluster.is_cluster:
        def cluster_preprocess_fn(x, cond):
            cluster_cond, cluster_x = utils.cluster(cond, cfg.cluster.num_clusters, verbose=cfg.cluster.verbose, placements=x)
            return cluster_x, cluster_cond
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        preprocess_fns.append(cluster_preprocess_fn)
        postprocess_fns.append(cluster_postprocess_fn)
    elif cfg.cluster.cached_clusters:
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        postprocess_fns.append(cluster_postprocess_fn)
    if cfg.sc_halo != 1.0:
        def resize_standard_cells(x, cond):
            _, _, sc_mask = analysis_utils.get_masks(x, cond)
            is_resize = sc_mask.float()
            size_multiplier = (is_resize * cfg.sc_halo) + ((1-is_resize))
            cond.x = cond.x * size_multiplier.unsqueeze(dim=-1)
            return x, cond
        preprocess_fns.append(resize_standard_cells)
    if cfg.edge_dropout > 0.0: # used for debugging
        def edge_dropout(x, cond):
            x, cond = utils.edge_dropout(x, cond, cfg.edge_dropout)
            return x, cond
        preprocess_fns.append(edge_dropout)
    if cfg.macros_only:
        if cfg.cached_macros:
            postprocess_fns.append(utils.add_non_macros)
        else:
            preprocess_fns.append(utils.remove_non_macros)
            postprocess_fns.append(utils.add_non_macros)
    def preprocess_fn(x, cond):
        for preprocess_step in preprocess_fns:
            x, cond = preprocess_step(x, cond)
        return x, cond
    def postprocess_fn(x, cond):
        for i, postprocess_step in enumerate(reversed(postprocess_fns)):
            x, cond = postprocess_step(x, cond)    
        return x, cond

    # Preparing dataset for train/val
    train_set, val_set = utils.load_graph_data(cfg.task, augment = cfg.augment, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = train_set[0][0].shape
    dataloader = utils.GraphDataLoader(train_set, val_set, cfg.batch_size, cfg.val_batch_size, device)
    # Preparing dataset for test
    _, test_set = utils.load_graph_data_with_config(cfg.test_task, train_data_limit = 0, val_data_limit = cfg.test_num_output_samples)

    with open_dict(cfg):
        if cfg.family in ["flow_matching",]:
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })

        else:
            raise NotImplementedError

    # Preparing model, optimizer, and grad scaler (for AMP)
    model_types = {
        "flow_matching": models.FlowMatchingModel,

    }
    if cfg.implementation == "custom":
        model = model_types[cfg.family](**cfg.model).to(device)
    else:
        raise NotImplementedError
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    grad_scaler = torch.cuda.amp.GradScaler(enabled = (device == "cuda"))
    train_metrics = common.Metrics()

    # Prepare logger
    num_params = sum([param.numel() for param in model.parameters()])
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))

    # Load checkpoint if exists
    print(OmegaConf.to_yaml(cfg))
    print(f"model has {num_params} params")
    load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler)

    # Start training
    print(f"==== Start Training on Device: {device} ====")
    model.train()
    t_0 = time.time()
    t_1 = time.time()
    best_loss = 1e12
    while step < cfg.train_steps:
        x, cond, idx = dataloader.get_batch("train")
        # x has (B, N, 2); netlist_data is a single graph in tg.Data format
        t = torch.randint(1, cfg.model.max_diffusion_steps + 1, [x.shape[0]], device = device)
        optim.zero_grad()

        loss, model_metrics = model.loss(x, cond, t, idx = idx, split = "train")

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optim)
        grad_scaler.update()

        train_metrics.add({"loss": loss.detach().cpu().item()})
        train_metrics.add(model_metrics)
        logger.add(train_metrics.result())
        step.increment()

        if (int(step)) % cfg.print_every == 0:
            t_2 = time.time()
            x_val, cond_val, idx_val = dataloader.get_batch("val")
            train_logs = utils.validate(x, model, cond)
            val_logs = utils.validate(x_val, model, cond_val)

            logger.add({
                "time_elapsed": t_2-t_0, 
                "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                })
            #logger.add(train_metrics.result())
            logger.add(val_logs, prefix="val")
            logger.add(train_logs, prefix="train")

            # display example images
            for split in ["train", "val"]:
                x_disp, cond_disp, idx_disp = dataloader.get_display_batch(cfg.display_examples, split = split)
                utils.display_graph_samples(cfg.display_examples, x_disp, cond_disp, model, logger, prefix = split)
                utils.display_forward_graph_samples(x_disp, cond_disp, model, logger, prefix = split)
            logger.write()
            t_1 = t_2

            checkpointer.save() # save latest checkpoint
            if val_logs["loss"] < best_loss:
                best_loss = val_logs["loss"]
                checkpointer.save(os.path.join(log_dir, "best.ckpt"))
                print("saving best model")
            cond_val.to(device="cpu")

        if (cfg.eval_every > 0) and ((int(step)) % cfg.eval_every == 0 or int(step) == 1):
            print(f"saving model at step {int(step)}")
            checkpointer.save(os.path.join(log_dir, f"step_{int(step)}.ckpt"))
            print("generating evaluation report")
            t3 = time.time()
            utils.generate_report(cfg.eval_samples, dataloader, model, logger, policy = cfg.eval_policy)
            logger.write()
            t4 = time.time()
            print(f"generated report in {t4-t3:.3f} sec")
        
        if (cfg.get("test_every", 0) > 0) and ((int(step)) % cfg.test_every == 0 or int(step) == 1):
            print(f"generating test report at step {int(step)}")
            t3 = time.time()

            test_sample_dir = os.path.join(sample_dir, "test")
            os.makedirs(test_sample_dir, exist_ok=True)
            output_metrics = {}
            log_metrics = common.Metrics()
            t_sample_start = time.time()
            for i in range(cfg.test_num_output_samples):
                x, cond = test_set[i]
                metrics, metrics_special, image, image_legalized = utils.save_outputs(
                    x, 
                    cond, 
                    model, 
                    save_folder=test_sample_dir, 
                    output_number_offset=i, 
                    policy=cfg.test_policy_algorithm,
                    policy_kwargs=cfg.test_policy,
                    preprocess_fn=preprocess_fn,
                    postprocess_fn=postprocess_fn,
                    legalization_fn=legalize_fn,
                )
                print(f"Finished test sample {i+1} of {cfg.test_num_output_samples} \t {metrics}")
                t_sample = time.time()
                # additional metrics
                eig_vals = analysis_utils.get_spectral_info(x, cond, k=1)
                metrics.update({
                    "num_vertices": x.shape[0],
                    "num_edges": cond.edge_index.shape[1],
                    "lambda_2": eig_vals[0],
                })
                logger.add({
                    "test_reverse_samples": {
                        **metrics,
                        **metrics_special,
                        "image": wandb.Image(image_legalized),
                        "image_raw": wandb.Image(image),
                        "time_elapsed": t_sample - t_sample_start,
                    }
                })
                # update metrics
                for k, v in metrics.items():
                    if k in output_metrics:
                        output_metrics[k].append(v)
                    else:
                        output_metrics[k] = [v]
                log_metrics.add(metrics)
            utils.dict_to_csv(output_metrics, os.path.join(log_dir,"test_metrics.csv"))
            for plot_keys in cfg.scatter_plots:
                x_name = plot_keys[0]
                y_name = plot_keys[1]
                if x_name in output_metrics and y_name in output_metrics:
                    scatter_plot = utils.plot_scatter(output_metrics[x_name], output_metrics[y_name], x_title=x_name, y_title=y_name)
                    logger.add({f"test_{x_name}_vs_{y_name}": scatter_plot})
            logger.add(log_metrics.result(), prefix="test")
            logger.add(cost(output_metrics), prefix = "test_sweep")
            logger.write()
            t4 = time.time()
            print(f"generated test report in {t4-t3:.3f} sec")

        cond.to(device="cpu")

def load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler):
    checkpointer.register({
            "step": step,
            "model": model,
            "optim": optim,
            "grad_scaler": grad_scaler,
        })
    if cfg.mode == "train":
        checkpointer.load(
            path_override = None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
            else os.path.join(cfg.log_dir, cfg.from_checkpoint)
        )
    elif cfg.mode in ["finetune", "ddpo"]:
        # Try to resume existing run
        loaded = checkpointer.load()
        if not loaded:
            # No existing run, so load pre-trained model only
            loaded = checkpointer.load(
                path_override = os.path.join(cfg.log_dir, cfg.from_checkpoint),
                filter_keys = ["model"],
            )
            if not loaded:
                print("WARNING Failed to load checkpoint for finetuning. Training from scratch instead.")
    else:
        raise NotImplementedError

if __name__=="__main__":
    main()