import os
import time
from datetime import datetime

import yaml
from ray.rllib.models import ModelCatalog

from agent.dqn import DQNAgent
from env.utils_v1 import ROOT_DIR, dict_update
from rl_module.action_mask_models import ActionMaskQModel

ModelCatalog.register_custom_model("action_mask_q", ActionMaskQModel)

LOG = False

if __name__ == "__main__":
    start = time.time()
    mdp_config = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/mdp_v16.yaml'), 'r'))
    dqn_config = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/dqn_v16.yaml'), 'r'))
    config = dict_update(mdp_config, dqn_config)
    dqn = DQNAgent(config=config, log_file=os.path.join(ROOT_DIR, 'log/runner_dqn.log'), version="v16")

    timestamp = datetime.now().strftime('%m%d_%H%M')
    dqn.test(timestamp=timestamp, duration=25, steps_per_map=1, log=LOG, suffix='before')
    # train the agent and evaluate every some steps
    dqn.train_and_eval(log=LOG, timestamp=timestamp)
    # tuner = tune.Tuner(
    #     "DQN",
    #     param_space=dqn_config.to_dict(),
    #     run_config=air.RunConfig(
    #         stop=config_run_train["stop"]
    #     ),
    # )
    # tuner.fit()

    # plot the action (TX location) of the trained agent vs. the optimal TX location
    # dqn.agent.restore("./checkpoint/dqn_0220_1342")
    dqn.test(timestamp=timestamp, duration=25, steps_per_map=1, log=LOG, suffix='after')

    end = time.time()
    print(f"total runtime: {end - start}s")
