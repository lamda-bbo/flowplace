PYTHONPATH=. python flow_matching/train_graph_with_test.py \
    method=train_mask_flowplace_uniform \
    task=mask.61 \
    print_every=100000 \
    eval_every=100000 \
    test_every=50000 \
    train_steps=1000000 \
    model.prior_noise_dist=uniform \
    test_task=iccad2015-s0 \
    test_num_output_samples=8


# If you do not want to test the model's performance on ICCAD2015 during training, set *test_every* to 0 and *test_task* to None.
