import os
import sys
sys.path.append('../')
from training.prediction import NeuralNetwork, Linear, GraphNeuralNetwork, GraphNeuralNetwork_sparse
import tensorflow as tf


def set_additional_args(args):
    args.predictor = args.strategy

    #if args.experiment is None:
    #    raise Exception("We must define experiment")
    if args.strategy in ["NeuralNetwork", "Linear", "GraphNeuralNetwork", "GraphNeuralNetwork_sparse"]:
        #if args.experiment == "experiment-0" and args.strategy == "NeuralNetwork":
        #    args.result_directory = args.result_directory + "NeuralNetwork_experiment-0_all_20/" + args.instance.split("/")[-1]
        #elif args.experiment == "experiment-7" and args.strategy == "NeuralNetwork":
        #    args.result_directory = args.result_directory + f"NeuralNetwork_experiment-7_{args.time_limit}_all_20/" + args.instance.split("/")[-1]
        #else:
        args.result_directory = args.result_directory + args.model_name + f"_timelimit-{args.time_limit}" + "/" + args.instance.split("/")[-1]
        args.model_name = args.model_directory + args.model_name
        #assert (args.experiment in args.model_name) or (args.experiment in ["experiment-0", "experiment-7"])
    else:
        args.result_directory = args.result_directory + args.strategy + f"_timelimit-{args.time_limit}" + (("_samplingrounds-" + str(args.monte_carlo_sampling_rounds)) if args.strategy == "monte_carlo" else "") + "/" + args.instance.split("/")[-1]

    args.result_directory = args.result_directory + f"_seed:{args.instance_seed}"

    # identify feature set
    if args.strategy in ["NeuralNetwork", "Linear", "GraphNeuralNetwork", "GraphNeuralNetwork_sparse"]:
        if ("static" in args.model_name) and ("dynamic" in args.model_name):
            args.feature_set = "dynamicstatic"
        elif "static" in args.model_name:
            args.feature_set = "static"
        elif "dynamic" in args.model_name:
            args.feature_set = "dynamic"
        else:
            raise Exception("Can not identify feature set.")

    # set additional time to monte-carlo benchmark
    if args.strategy == "monte_carlo":
        args.time_limit = args.time_limit * args.monte_carlo_sampling_rounds

    # set sampling rounds for rolling-horizon benchmark
    args.monte_carlo_sampling_rounds = 1 if args.strategy != "monte_carlo" else args.monte_carlo_sampling_rounds

    return args


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("New dir {} ... created".format(directory))


def create_directories(list_directories):
    for directory in list_directories:
        create_directory(directory)


def load_SL_model(args, directory):
    if args.strategy == "NeuralNetwork":
        model = NeuralNetwork()
        model.model = tf.keras.models.load_model(directory)
        return model
    elif args.strategy == "Linear":
        model = Linear()
        model.model = tf.keras.models.load_model(directory)
        return model
    elif args.strategy == "GraphNeuralNetwork":
        model = GraphNeuralNetwork()
        model.model = tf.keras.models.load_model(directory)
        return model
    elif args.strategy == "GraphNeuralNetwork_sparse":
        model = GraphNeuralNetwork_sparse()
        model.model = tf.keras.models.load_model(directory)
        return model


def make_to_ints(profits):
    return profits.astype(int)