import time
import yaml

from agent.dqn import DQNAgent


if __name__ == "__main__":
    start = time.time()
    config = yaml.safe_load(open('../config/dqn_test.yaml', 'r'))
    dqn = DQNAgent(config=config, log_file='../log/runner_dqn.log')

    # train the agent and evaluate every some steps
    dqn.train_and_eval()
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
    dqn.test()

    end = time.time()
    print(f"total runtime: {end - start}s")
