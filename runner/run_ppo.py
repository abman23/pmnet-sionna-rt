import os
import time
import yaml
from torch.distributions import Distribution

from agent.ppo import PPOAgent
from env.utils_v1 import ROOT_DIR

if __name__ == "__main__":
    # prevent error caused by simplex check failure
    Distribution.set_default_validate_args(False)

    start = time.time()
    config_ppo = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/ppo_v1.yaml'), 'r'))
    ppo = PPOAgent(config=config_ppo, log_file=os.path.join(ROOT_DIR, 'log/runner_ppo.log'), version="v13")

    # ppo.param_tuning(
    #     # lr_schedule=[1e-4, 5e-5, 1e-5],
    #     # bs_schedule=[32, 64, 128],
    #     gamma_schedule=[0.1, 0.],
    #     training_iteration=100)

    # train the agent and evaluate every some steps
    ppo.train_and_eval(log=True)

    # plot the action (TX location) of the trained agent vs. the optimal TX location
    # ppo.agent.restore("./checkpoint/ppo_0225_2322")
    ppo.test(duration=5, log=True)

    end = time.time()
    print(f"total runtime: {end - start}s")
