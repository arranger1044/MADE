ipython -- grid_search.py nltcs --learning_rate 1e-5 1e-7 1e-9 --decrease_constant 0.95 --max_epochs -1 --shuffle_mask -1 --shuffling_type Full --nb_shuffle_per_valid 300 --batch_size 100 --look_ahead 30 --pre_training False --pre_training_max_epoc 0 --update_rule adadelta --dropout_rate 0 --hidden_sizes "[500]" "[500,500]" --random_seed 1234 --use_cond_mask False True --direct_input_connect Output --direct_output_connect False --hidden_activation hinge softplus --weights_initialization Orthogonal --mask_distribution 0 | parallel -j 2