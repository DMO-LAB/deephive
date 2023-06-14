from src.environment import OptimizationEnv
from src.mappo import MAPPO
from src.deephive import DeepHive
from distutils.util import strtobool
import os
import argparse
import numpy as np
from commons.utils import get_config, get_args, get_environment, get_policy


def main(title, env, policy, mode, config, **kwargs):
    deephive = DeepHive(title, env, policy, mode, config, **kwargs)
    deephive.optimize(debug=True)

if __name__ == '__main__':
    args = get_args()
    title = args.title
    env_name = args.env
    mode = args.mode
    reinit = args.reinit
    log = args.log
    tags = args.tags
    config = get_config(args.config)
    # ENVIRONMENT
    env = get_environment(config, env_name, reinit=reinit)
    # POLICY
    policy = get_policy(config, mode)
    # MAIN
    main(title, env, policy, mode=mode, config=config, log=log, tags=tags)


