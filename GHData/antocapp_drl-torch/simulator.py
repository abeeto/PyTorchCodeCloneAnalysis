#handles train and testing at large scale
#demo
from configs import cfg
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        dest="config", 
        required=True
        )
    args = parser.parse_args()
    return args 

def main(config_file):

    cfg.merge_from_file(config_file)
    if cfg.TEST.TEST_ONLY:
        cfg.TRAIN.TRAIN = False
        cfg.NETWORK.PARAMS.write = False
        cfg.NETWORK.PARAMS.test_only = True
    cfg.freeze()

    networks = __import__("networks")
    network_ = getattr(networks, cfg.NETWORK.ARCH)

    engine = __import__("engine")
    environment_ = getattr(engine, cfg.ENVIRONMENT.NAME)


    #create the agent
    agent = network_(**cfg.NETWORK.PARAMS)
    
    if cfg.TRAIN.TRAIN:

        if cfg.TRAIN.ASYNCH:
            from utils.parallels import SubprocVecEnv

            #generate subprocesses for envs
            environment = [environment_() for _ in range(cfg.TRAIN.WORKERS)]
            environment = SubprocVecEnv(environment)

            if cfg.TRAIN.TEST:
                #generate one instance of environment for testing
                agent.test_env = environment_() 
        else:
            environment = environment_()

        #specification for training
        specs = cfg.TRAIN.SPECS
        agent.trainer(environment, **specs)
        return "Training completed!"

    elif cfg.TEST.TEST_ONLY:
        environment = environment_()
        cumulative_reward = agent.test(environment, cfg.TEST.NUMBER)
        return "Testing completed! Cumulative reward: {}"\
        .format(cumulative_reward)


if __name__ == "__main__":
    a = parse()
    config_file = a.config
    main(config_file)