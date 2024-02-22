import json
import yaml

# miscellaneous script
if __name__ == '__main__':
    # json.dump(config_run_train, open('config/dqn_train.json', 'w'), indent=4)
    # json.dump(config_run_test, open('config/dqn_test.json', 'w'), indent=4)
    # json.dump(config_run_train, open('config/ppo_train.json', 'w'), indent=4)

    config_json = json.load(open('config/dqn_test.json', 'r'))
    yaml.dump(config_json, open('config/dqn_test.yaml', 'w'))
