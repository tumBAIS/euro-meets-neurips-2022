import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, help="Strategy to use", choices=("oracle", "greedy", "random", "lazy", "NeuralNetwork",
                                                                             "Linear", "GraphNeuralNetwork", "GraphNeuralNetwork_sparse",
                                                                             "rolling_horizon", "monte_carlo"), default="NeuralNetwork")
parser.add_argument("--result_directory", type=str, help="specify directory to save results", default="./results/results_testing/")
parser.add_argument("--model_name", type=str, default="NeuralNetwork_samples-50_instances-15_runtime-3600_featureset-dynamicstatic_iteration-29")
parser.add_argument("--instance", help="Instance to solve", default="../instances/ORTEC-VRPTW-ASYM-db776df3-d1-n322-k23.txt")
parser.add_argument("--instance_seed", type=int, default=2000, help="Seed to use for the dynamic instance")
parser.add_argument("--time_limit", type=int, default=20, help="Time limit in seconds")
parser.add_argument("--solver_seed", type=int, default=1, help="Seed to use for the solver")
parser.add_argument("--sample_seed", type=int, default=1234, help="Seed for sampling in rolling horizon and monte-carlo")
parser.add_argument("--monte_carlo_sampling_rounds", type=int, default=2, help="Number of sampling rounds for monte-carlo benchmark")
parser.add_argument("--instance_directory", type=str, help="specify directory with instances to run", default=None)
parser.add_argument("--model_directory", type=str, help="specify directory to save model", default="../training/models/")

