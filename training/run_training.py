import numpy as np
import random
from pathlib import Path
import multiprocessing as mp
from src import config, util
from features.FeatureComputer import create_features
from optimization import Optimizer

def get_model_name(args):
    oracle_solution_directory = args.oracle_solutions_directory
    oracle_solution_directory = oracle_solution_directory.split("/")[2]
    args.model_name = args.predictor + "_" + oracle_solution_directory + "_featureset-" + args.feature_set
    return args


if __name__ == "__main__":

    print("Number available CPU: {}".format(mp.cpu_count()))

    args = config.parser.parse_args()
    args.pchgs_executable = str(next(Path("../pchgs/build").rglob("PCHGS*")))
    args = get_model_name(args)
    util.create_directories([args.dir_models])
    print(args)

    random.seed(123)
    np_random_seed = np.random.seed(123)

    # load training instances
    training_instances = util.load_training_instances(args)

    # calculate features for training instances
    training_instances = create_features(args, training_instances)

    # initalize optimizer
    optim = Optimizer(args=args, num_features=training_instances[0]["features"].shape[1], num_edge_features=training_instances[0]["edge_features"].shape[1])

    # optimizer trains ML-CO policy
    optim.train(training_instances)
