config_run = {
    "env": {
        "map_size": 64,  # number of rows and columns of the cropped pixel map
        "ratio_coverage": .05,  # ratio of pixels counted as 'covered' when calculating the path loss threshold
        "max_steps": 100,  # maximum number of iteration steps before terminating the environment
        "n_cropped": 100,  # number of cropped maps
        "no_masking": True,  # true if not use action masking, otherwise false
        # the dict containing coefficients for each reward/penalty function
        "coefficient_dict": {
            "r_c": .1,
            "p_d": 1.,
            "p_b": 1.
        }
    },
    "train": {
        "train_batch_size": 100
    },
    "eval": {
        "evaluation_interval": 5,
        "evaluation_duration": 1
    },
    "stop": {
        "training_iteration": 2,
    }
}

config_run2 = {
    "env": {
        "map_size": 256,  # number of rows and columns of the cropped pixel map
        "ratio_coverage": .2,  # ratio of pixels counted as 'covered' when calculating the path loss threshold
        "max_steps": 100,  # maximum number of iteration steps before terminating the environment
        "n_cropped": 50,  # number of cropped maps
        "no_masking": True,  # true if not use action masking, otherwise false
        # the dict containing coefficients for each reward/penalty function
        "coefficient_dict": {
            "r_c": .1,
            "p_d": 1.,
            "p_b": 1.
        }
    },
    "train": {
        "train_batch_size": 100
    },
    "eval": {
        "evaluation_interval": 5,
        "evaluation_duration": 1
    },
    "stop": {
        "training_iteration": 2,
    }
}
