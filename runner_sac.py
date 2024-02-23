import time
import yaml

from agent.sac import SACAgent


if __name__ == "__main__":
    start = time.time()
    config = yaml.safe_load(open('config/sac_train.yaml', 'r'))
    # print(config)
    sac = SACAgent(config=config, log_file='log/runner_sac.log')

    # train the agent and evaluate every some steps
    # sac.train_and_eval()
    # tuner = tune.Tuner(
    #     "DQN",
    #     param_space=dqn_config.to_dict(),
    #     run_config=air.RunConfig(
    #         stop=config_run_train["stop"]
    #     ),
    # )
    # tuner.fit()

    # plot the action (TX location) of the trained agent vs. the optimal TX location
    sac.agent.restore("./checkpoint/sac_0222_0617")
    sac.eval_plot()

    end = time.time()
    print(f"total runtime: {end - start}s")
