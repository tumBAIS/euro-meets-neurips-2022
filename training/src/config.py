import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--oracle_solutions_directory", type=str, default="../experiments/samples-test_instances-test_runtime-test/")
parser.add_argument("--instances_directory", type=str, default="../instances/")
parser.add_argument("--predictor", type=str, choices=["NeuralNetwork", "GraphNeuralNetwork", "Linear", "GraphNeuralNetwork_sparse"], default="NeuralNetwork")
parser.add_argument("--num_training_epochs", type=int, help="number of training epochs", default=50)
parser.add_argument("--time_limit", type=int, default=20, help="Time limit in seconds")
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--dir_models", type=str, help="directory to save models and normalizations", default="./models/")
parser.add_argument("--num_perturbations", type=int, help="number of perturbations to calculate the SL loss", default=2)
parser.add_argument("--sd_perturbation", type=float, default=1)
parser.add_argument("--feature_set", type=str, choices=["static", "dynamic", "dynamicstatic"], default="dynamicstatic")