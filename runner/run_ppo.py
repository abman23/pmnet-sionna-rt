import os
import time
from datetime import datetime

import yaml
from torch.distributions import Distribution

# from agent.ppo import PPOAgent
from multi_agent.async_ppo import AsyncPPO
from env.utils_v1 import ROOT_DIR

LOG = True
VERSION = 'v32'

if __name__ == "__main__":
    # prevent error caused by simplex check failure
    Distribution.set_default_validate_args(False)

    start = time.time()
    config_ppo = yaml.safe_load(open(os.path.join(ROOT_DIR, f'config/mdp_{VERSION}.yaml'), 'r'))
    ppo = AsyncPPO(config=config_ppo, log_file=os.path.join(ROOT_DIR, f'log/runner_ppo_{VERSION}.log'), version=VERSION)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    # timestamp = '0424_1832'
    # ppo.test(timestamp=timestamp, duration=50, steps_per_map=1, log=LOG, suffix='before')
    # ppo.param_tuning(
    #     # lr_schedule=[1e-4, 5e-5, 1e-5],
    #     # bs_schedule=[32, 64, 128],
    #     gamma_schedule=[0.1, 0.],
    #     training_iteration=100)

    # train the agent and evaluate every some steps
    ppo.train_and_eval(log=LOG, timestamp=timestamp)

    # ppo.continue_train(start_episode=200, data_path=os.path.join(ROOT_DIR, 'data/v30_ppo_0423_0127.json'),
    #                    model_path=os.path.join(ROOT_DIR, 'checkpoint/v30_ppo_0423_0127'), log=LOG, timestamp=timestamp)

    # ppo.agent.restore(os.path.join(ROOT_DIR, 'checkpoint', f'{VERSION}_ppo_{timestamp}'))
    ppo.test(timestamp=timestamp, duration=50, log=LOG, suffix='after', test_on_trained=False)

    end = time.time()
    print(f"total runtime: {end - start}s")
