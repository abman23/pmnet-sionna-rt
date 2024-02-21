config_run_test = {
    "env": {
        "cropped_map_size": 64,  # number of rows and columns of the cropped pixel map
        "ratio_coverage": .2/16,  # ratio of pixels counted as 'covered' when calculating the path loss threshold
        "original_map_path": "./resource/usc.png",
        "original_map_scale": 880/256,
        # "max_steps": 10000,  # maximum number of iteration steps before terminating the environment
        "n_maps": 100,  # number of cropped resource
        "n_steps_per_map": 10,
        "no_masking": True,  # true if not use action masking, otherwise false
        # the dict containing coefficients for each reward/penalty function
        "coefficient_dict": {
            "r_c": 1.,
            "p_d": 1.,
            "p_b": 1.
        },
        "preset_map_path": "./resource/setup.json",
    },
    "explore": {
        "explore": True,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 300,
        }
    },
    "train": {
        "train_batch_size": 8,
        # "lr": 1e-3,
        # "gamma": 0.9,
        "num_steps_sampled_before_learning_starts": 100,
        "replay_buffer_config": {"capacity": 50000},
    },
    "eval": {
        "evaluation_interval": 5,
        "evaluation_duration": 3,
        "evaluation_config": {
            "explore": False,
            "env_config": {"evaluation": True},
        },
    },
    "report": {
        "min_sample_timesteps_per_iteration": 100,
    },
    "stop": {
        "training_iteration": 10,
    },
    "rollout": {
        "num_rollout_workers": 4,
        "num_envs_per_worker": 1,
    },
    "resource": {
        "num_gpus": 0,
        "num_cpus_per_worker": 2,
        "num_gpus_per_worker": 0,
    },
}

config_run_train = {
    "env": {
        "cropped_map_size": 64,  # number of rows and columns of the cropped pixel map
        "ratio_coverage": .2/16,  # ratio of pixels counted as 'covered' when calculating the path loss threshold
        "original_map_path": "./resource/usc.png",
        "original_map_scale": 880/256,
        "n_maps": 100,  # number of cropped resource
        "n_steps_per_map": 50,
        "no_masking": True,  # true if not use action masking, otherwise false
        # the dict containing coefficients for each reward/penalty function
        "coefficient_dict": {
            "r_c": 1.,
            "p_d": 1.,
            "p_b": 1.
        },
        "preset_map_path": "./resource/setup.json",
    },
    "explore": {
        "explore": True,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 100000,
        }
    },
    "train": {
        "train_batch_size": 32,
        "lr": 1e-4,
        # "gamma": 0.9,
        "num_steps_sampled_before_learning_starts": 50000,
        "replay_buffer_config": {"capacity": 50000},
    },
    "eval": {
        "evaluation_interval": 5,
        "evaluation_duration": 3,
        "evaluation_config": {
            "explore": False,
            "env_config": {"evaluation": True, "preset_map_path": None, "n_maps": 3},
        },
    },
    "report": {
        "min_sample_timesteps_per_iteration": 1000,
    },
    "stop": {
        "training_iteration": 1000,
    },
    "rollout": {
        "num_rollout_workers": 4,
        "num_envs_per_worker": 1,
    },
    "resource": {
        "num_gpus": 0,
        "num_cpus_per_worker": 2,
        "num_gpus_per_worker": 0,
    },
}

config_run_sac = {
    "env": {
        "cropped_map_size": 64,  # number of rows and columns of the cropped pixel map
        "ratio_coverage": .2/16,  # ratio of pixels counted as 'covered' when calculating the path loss threshold
        "original_map_path": "/Users/ylu/Documents/USC/WiDeS/BS_Deployment/resource/usc.png",
        "original_map_scale": 880/256,
        # "max_steps": 10000,  # maximum number of iteration steps before terminating the environment
        "n_maps": 10,  # number of cropped resource
        "n_steps_per_map": 100,
        "no_masking": True,  # true if not use action masking, otherwise false
        # the dict containing coefficients for each reward/penalty function
        "coefficient_dict": {
            "r_c": 1.,
            "p_d": 1.,
            "p_b": 1.
        }
    },
    "explore": {
        "explore": True,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 20000,
        }
    },
    "train": {
        "train_batch_size": 32,
        # "lr": 1e-3,
        # "gamma": 0.9,
        "num_steps_sampled_before_learning_starts": 10000,
        "replay_buffer_config": {"capacity": 20000},
        "target_network_update_freq": 5,
    },
    "eval": {
        "evaluation_interval": 1,
        "evaluation_duration": 3,
        "evaluation_config": {"explore": False},
    },
    "report": {
        "min_sample_timesteps_per_iteration": 1000,
    },
    "stop": {
        "training_iteration": 20,
    }
}
