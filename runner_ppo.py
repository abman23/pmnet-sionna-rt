import time
import yaml

from agent.ppo import PPOAgent


if __name__ == "__main__":
    start = time.time()
    config_ppo = yaml.safe_load(open('config/ppo_train.yaml', 'r'))
    ppo = PPOAgent(config=config_ppo, log_file='log/runner_ppo.log')

    # train the agent and evaluate every some steps
    ppo.train_and_eval(log=True)

    # plot the action (TX location) of the trained agent vs. the optimal TX location
    # ppo.agent.restore("./checkpoint/ppo_0225_2322")
    ppo.eval_plot(log=True)

    end = time.time()
    print(f"total runtime: {end - start}s")
