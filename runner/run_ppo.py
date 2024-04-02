import os
import time
from datetime import datetime

import yaml
from torch.distributions import Distribution

from agent.ppo import PPOAgent
from env.utils_v1 import ROOT_DIR

LOG = True

if __name__ == "__main__":
    # prevent error caused by simplex check failure
    Distribution.set_default_validate_args(False)

    start = time.time()
    config_ppo = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/mdp_v16.yaml'), 'r'))
    ppo = PPOAgent(config=config_ppo, log_file=os.path.join(ROOT_DIR, 'log/runner_ppo.log'), version="v16")

    timestamp = datetime.now().strftime('%m%d_%H%M')
    ppo.test(timestamp=timestamp, duration=50, steps_per_map=1, log=LOG, suffix='before')
    # ppo.param_tuning(
    #     # lr_schedule=[1e-4, 5e-5, 1e-5],
    #     # bs_schedule=[32, 64, 128],
    #     gamma_schedule=[0.1, 0.],
    #     training_iteration=100)

    # train the agent and evaluate every some steps
    ppo.train_and_eval(log=LOG, timestamp=timestamp)

    # ppo.continue_train(start_episode=350, data_path=os.path.join(ROOT_DIR, 'data/ppo_0331_1940.json'),
    #                    model_path=os.path.join(ROOT_DIR, 'checkpoint/ppo_0330_1940'), log=LOG, timestamp=timestamp)
    ppo.test(timestamp=timestamp, duration=50, steps_per_map=1, log=LOG, suffix='after')

    end = time.time()
    print(f"total runtime: {end - start}s")
