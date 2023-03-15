import os
import sys
sys.path.append('../')
import json
from collections import OrderedDict
from evaluation import tools
from evaluation.environment import VRPEnvironment
import numpy as np


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("New dir {} ... created".format(directory))


def create_directories(list_directories):
    for directory in list_directories:
        create_directory(directory)


def get_observation_and_solution(args, training_instance_directory):
    instance = tools.read_vrplib(args.instances_directory + training_instance_directory.split("_")[0] + ".txt")
    oracle_solution = json.load(open(os.path.join(args.oracle_solutions_directory + training_instance_directory + "/best-sol.json"), "r"))[0]["routes"]
    seed = json.load(open(os.path.join(args.oracle_solutions_directory + training_instance_directory + "/seeds.json"), "r"))

    # we iterate over all epoch observations and save solutions
    epoch_observations = OrderedDict()
    for observation_file in list(os.listdir(args.oracle_solutions_directory + training_instance_directory)):
        if "observation" not in observation_file:
            continue
        current_epoch = int(observation_file.split("_")[-1].split(".")[0])
        epoch_observations[current_epoch] = json.load(open(os.path.join(args.oracle_solutions_directory + training_instance_directory, observation_file), "r"))
        # convert data to numpy
        for key in epoch_observations[current_epoch]["epoch_instance"]:
            if key == "capacity":
                continue
            epoch_observations[current_epoch]["epoch_instance"][key] = np.array(epoch_observations[current_epoch]["epoch_instance"][key])
    epoch_observations = OrderedDict(sorted(epoch_observations.items()))

    # we iterate over all solutions and save epoch solutions
    epoch_solutions = {key: [] for key in epoch_observations}
    for route in oracle_solution:
        epoch_solutions[route["epoch"]].append(route["customers"])

    # we get static info
    env = VRPEnvironment()
    _, static_info = env.reset(seed=seed["instance_seed"], instance=instance)

    return {"epoch_observations": [observation for observation in epoch_observations.values()][:-1],  # we can ignore the last epoch as there are all must-dispatch
            "epoch_solutions": [solution for solution in epoch_solutions.values()][:-1],  # we can ignore the last epoch as there are all must-dispatch
            "instance": instance, "static_info": static_info}



def get_num_observations_avg(training_instances_collecter):
    num_training_instances = len(training_instances_collecter)
    num_epoch_observations = 0
    for training_instance in training_instances_collecter:
        num_epoch_observations += len(training_instance["epoch_observations"])
    print(f"Num epoch observations: {num_epoch_observations}, on average {num_epoch_observations/num_training_instances} observations / training instance")



def load_training_instances(args):
    training_instances_directories = list(os.listdir(args.oracle_solutions_directory))
    training_instances_collecter = []
    for training_instance_directory in training_instances_directories:
        if os.path.isfile(os.path.join(args.oracle_solutions_directory + training_instance_directory + "/best-sol.json")):
            training_instances_collecter.append(get_observation_and_solution(args, training_instance_directory))
    get_num_observations_avg(training_instances_collecter)
    return training_instances_collecter


def calculate_default_accuracy(training_instances):
    target_n = []
    for training_instance in training_instances:
        target_n += training_instance["epoch_instance"]["target"].tolist()
    return calculate_accuracy_mean(n=np.array(target_n), n_hat=np.zeros(len(target_n)))


def calculate_accuracy_mean(n, n_hat):
    return 1 - (np.sum(np.logical_xor(n, n_hat)) / n_hat.size)


def decode_solution(solution, edges, nodes):
    if isinstance(solution, dict):  # the solution is a pchgs solution
        cost = solution["cost"]
        if solution["routes"] is None:
            solution = [[]]
        else:
            solution = [route["requests"] for route in solution["routes"]]
    else:  # the solution is a list of routes
        cost = None
    edges_in_solution = get_edges_in_solution(solution=solution)
    y = [edge in edges_in_solution for edge in edges]
    nodes_in_solution = get_nodes_in_solution(solution=solution)
    n = [node in nodes_in_solution for node in nodes]
    return cost, y, n


def get_nodes_in_solution(solution):
    nodes = [node for node_list in solution for node in node_list]
    return list(set(nodes))


def get_edges_in_solution(solution):
    edges_solution = []
    for solution_route in solution:
        for request_idx, request in enumerate(solution_route):
            if request_idx == 0:  # add first element
                edges_solution.append([0, request])
            if request_idx == len(solution_route) - 1:
                edges_solution.append([request, 0])  # add last element
            else:
                edges_solution.append([request, solution_route[request_idx + 1]])
    return edges_solution


def apply_perturbation(args, profits):
    profits_perturbed = profits + np.random.normal(loc=0, scale=args.sd_perturbation, size=profits.shape)
    profits_perturbed[0] = 0
    return np.squeeze(profits_perturbed)


def make_to_ints(profits):
    return profits.astype(int)



