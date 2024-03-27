import os
import time
import yaml

from agent.sac import SACAgent
from env.utils_v1 import ROOT_DIR

if __name__ == "__main__":
    start = time.time()
    config = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/sac_v1.yaml'), 'r'))
    # print(config)
    sac = SACAgent(config=config, log_file=os.path.join(ROOT_DIR, 'log/runner_ppo.log'), version="v14")

    # train the agent and evaluate every some steps
    timestamp = sac.train_and_eval(log=True)

    # plot the action (TX location) of the trained agent vs. the optimal TX location
    # sac.agent.restore("./checkpoint/sac_0222_0617")
    sac.test(timestamp=timestamp, duration=5, steps_per_map=10, log=True)

    end = time.time()
    print(f"total runtime: {end - start}s")
