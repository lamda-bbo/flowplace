PYTHONPATH=. python flow_matching/eval.py \
    method=eval_flow_matching \
    task=iccad2015-s0 \
    from_checkpoint=mask.61.train_mask_flowplace_uniform.61/best.ckpt \
    log_dir=logs/debug \
    model.prior_noise_dist=uniform \
    model.max_diffusion_steps=20 \
    model.use_hard_constraint_in_sampling=True \
    model.hard_constraint_w_reg=0.1 \
    model.hard_constraint_w_dist=1.0 \
    num_output_samples=8 \
    output_placement=True \
    logger.wandb=False

# *task*: [iccad2015-s0|openroad-s0|ispd2005-macros] specifies the benchmark task being evaluated; please make sure the corresponding graph data is available in `datasets/graph` and benchmark files are available.

# *from_checkpoint* should be set to the checkpoint path of the model you want to evaluate; if you have trained the model using the provided training script, you can find the checkpoint in the corresponding training log directory

# *prior_noise_dist* must match the setting used during training

# *max_diffusion_steps* is the number of sampling steps used during model inference; 10-50 is usually optimal

# *use_hard_constraint_in_sampling* means hard constraints are used during sampling.

# *hard_constraint_w_reg* and *hard_constraint_w_dist* are the preference weights used during legalization; 0.1 and 1.0 mean that, while considering the original position, hard-constraint correction slightly pushes macros toward areas closer to regularity

# Please make sure *num_output_samples* matches the number of benchmark tasks being evaluated