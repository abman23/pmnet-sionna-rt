import time
import yaml

from agent.ppo import PPOAgent


if __name__ == "__main__":
    start = time.time()
    config_ppo = yaml.safe_load(open('config/ppo_test.yaml', 'r'))
    ppo = PPOAgent(config=config_ppo, log_file='log/runner_ppo.log')

    # train the agent and evaluate every some steps
    ppo.train_and_eval(log=False)

    # plot the action (TX location) of the trained agent vs. the optimal TX location
    # dqn.agent.restore("./checkpoint/dqn_0220_1342")
    # ppo.eval_plot(log=False)

    end = time.time()
    print(f"total runtime: {end - start}s")
