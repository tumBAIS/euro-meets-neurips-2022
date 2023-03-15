import copy
import os
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
import seaborn as sns
import collections


def get_not_relevant_instances(relevant_directories):
    not_relevant_instances = []
    for relevant_directory in relevant_directories:
        for result_dict in os.listdir(relevant_directory):
            if not os.path.isfile(relevant_directory + "/" + result_dict + "/" + "best-sol.json"):
                if relevant_directory.split("/")[-1].split("_")[0] not in ["monte", "rolling", "greedy", "lazy", "random", "oracle"]:
                    print(relevant_directory + "/" + result_dict)
                not_relevant_instances.append(result_dict)
    return set(not_relevant_instances)


def fill_results_saver(relevant_directories, experiment, scheme, not_relevant_instances):
    ## FILL RESULTS SAVER
    results_saver = {}

    for result_dict in relevant_directories:
        model = get_model_name(result_dict)
        learning_iteration = get_iteration(result_dict)
        for dir_instance in [dir_instance for dir_instance in os.listdir(result_dict) if dir_instance not in not_relevant_instances]:
            best_sol = json.load(open(result_dict + "/" + dir_instance + "/" + "best-sol.json", "r"))
            seed = int(dir_instance.split("_seed:")[-1])
            dir_instance = dir_instance.split("_seed:")[0]
            if model not in results_saver.keys():
                results_saver[model] = {}
            if learning_iteration not in results_saver[model].keys():
                results_saver[model][learning_iteration] = {}
            if dir_instance not in results_saver[model][learning_iteration].keys():
                results_saver[model][learning_iteration][dir_instance] = {}
            if seed not in results_saver[model][learning_iteration][dir_instance].keys():
                results_saver[model][learning_iteration][dir_instance][seed] = []
            results_saver[model][learning_iteration][dir_instance][seed].append(best_sol[0]["cost"] if best_sol[0]["cost"]<0 else -1 * best_sol[0]["cost"])
    return results_saver


def sort(results_saver):
    # sort dictionary according to best performing model first
    results_saver = OrderedDict(sorted(results_saver.items(), key=lambda item: np.mean(list(item[1].values()))))
    for key in results_saver.keys():
        results_saver[key] = OrderedDict(sorted(results_saver[key].items(), key=lambda item: item[0]))
    return results_saver


def get_relative_best_values(results_saver):
    saver_best = {}
    for mode in results_saver.keys():
        for learning_iteration in results_saver[mode].keys():
            for instance in results_saver[mode][learning_iteration].keys():
                if instance not in saver_best.keys():
                    saver_best[instance] = results_saver[mode][learning_iteration][instance]
                if saver_best[instance] < results_saver[mode][learning_iteration][instance]:
                    saver_best[instance] = results_saver[mode][learning_iteration][instance]
    for mode in results_saver.keys():
        for learning_iteration in results_saver[mode].keys():
            results_saver[mode][learning_iteration] = {instance: results_saver[mode][learning_iteration][instance] - saver_best[instance] for instance in results_saver[mode][learning_iteration].keys()}
            results_saver[mode][learning_iteration] = [results_saver[mode][learning_iteration][instance] / saver_best[instance] for instance in results_saver[mode][learning_iteration].keys()]
            results_saver[mode][learning_iteration] = [100 * x for x in results_saver[mode][learning_iteration]]  # change to %
    return results_saver


def get_relative_greedy_values(results_saver):
    for mode in [mode for mode in results_saver.keys() if mode != "greedy"]:
        for learning_iteration in results_saver[mode].keys():
            results_saver[mode][learning_iteration] = {instance: results_saver[mode][learning_iteration][instance] - results_saver["greedy"][-1][instance] for instance in results_saver[mode][learning_iteration].keys()}
            results_saver[mode][learning_iteration] = [results_saver[mode][learning_iteration][instance] / results_saver["greedy"][-1][instance] for instance in results_saver[mode][learning_iteration].keys()]
            results_saver[mode][learning_iteration] = [100 * x for x in results_saver[mode][learning_iteration]]  # change to %
    del results_saver["greedy"]
    return results_saver


def get_relative_monte_carlo_values(results_saver):
    for mode in [mode for mode in results_saver.keys() if mode != "monte_carlo_experiment-0"]:
        for learning_iteration in results_saver[mode].keys():
            results_saver[mode][learning_iteration] = {instance: results_saver[mode][learning_iteration][instance] - results_saver["monte_carlo_experiment-0"][-1][instance] for instance in results_saver[mode][learning_iteration].keys()}
            results_saver[mode][learning_iteration] = [results_saver[mode][learning_iteration][instance] / results_saver["monte_carlo_experiment-0"][-1][instance] for instance in results_saver[mode][learning_iteration].keys()]
            results_saver[mode][learning_iteration] = [100 * x for x in results_saver[mode][learning_iteration]]  # change to %
    del results_saver["monte_carlo_experiment-0"]
    return results_saver


def delete_non_consistend_instances(results_saver):
    instance_set = None
    for key in results_saver.keys():
        for learning_iteration in results_saver[key].keys():
            if instance_set:
                instance_set = instance_set.intersection(set(results_saver[key][learning_iteration].keys()))
            else:
                instance_set = set(results_saver[key][learning_iteration].keys())
    for key in results_saver.keys():
        for learning_iteration in results_saver[key].keys():
            for instance in list(results_saver[key][learning_iteration].keys()):
                if instance not in instance_set:
                    del results_saver[key][learning_iteration][instance]
    return results_saver


def get_relative_oracle_values(results_saver):
    for mode in [mode for mode in results_saver.keys() if mode != "oracle_timelimit-3600"]:
        for learning_iteration in results_saver[mode].keys():
            results_saver[mode][learning_iteration] = {instance: results_saver[mode][learning_iteration][instance] - results_saver["oracle_timelimit-3600"][-1][instance] for instance in results_saver[mode][learning_iteration].keys()}
            results_saver[mode][learning_iteration] = [results_saver[mode][learning_iteration][instance] / results_saver["oracle_timelimit-3600"][-1][instance] for instance in results_saver[mode][learning_iteration].keys()]
            results_saver[mode][learning_iteration] = [100 * x for x in results_saver[mode][learning_iteration]]  # change to %
    del results_saver["oracle_timelimit-3600"]
    return results_saver


def get_absolute_values(results_saver):
    for mode in results_saver.keys():
        for learning_iteration in results_saver[mode].keys():
            results_saver[mode][learning_iteration] = list(results_saver[mode][learning_iteration].values())
    return results_saver


def get_model_per_instance_values(results_saver):
    original_keys = [key for key in list(results_saver.keys()) if "NeuralNetwork_experiment-6" in key]
    results_saver["NeuralNetworkInd_experiment-6"] = {-1: {}}
    for key in original_keys:
        for learning_iteration in results_saver[key].keys():
            results_saver["NeuralNetworkInd_experiment-6"][-1][key.split("_")[-2] + ".txt"] = results_saver[key][learning_iteration][key.split("_")[-2] + ".txt"]
    for key in original_keys:
        del results_saver[key]
    return results_saver


def clean_from_missing_values(results_saver):
    output_saver = OrderedDict()
    for key in results_saver.keys():
        output_saver[key] = OrderedDict()
        max_elements = max(len(i) for i in list(results_saver[key].values()))
        for iteration in results_saver[key].keys():
            if len(results_saver[key][iteration]) == max_elements:
                output_saver[key][iteration] = results_saver[key][iteration]
    return output_saver


def set_case(base_case, key, new_value):
    base_case_copy = copy.deepcopy(base_case)
    base_case_copy[key] = new_value
    return base_case_copy


base_case = {"strategy": "NeuralNetwork", "samples": 50, "instances": 15, "runtime": 3600, "feature_set": "dynamicstatic", "predictor": "NeuralNetwork", "timelimit": 90}

def get_strategy(result_directory):
    if "sparse" in result_directory:
        return "GraphNeuralNetwork_sparse"
    else:
        return result_directory.split("/")[-1].split("_")[0]

def get_samples(result_directory):
    return int(result_directory.split("/")[-1].split("samples-")[1].split("_")[0])

def get_instances(result_directory):
    return int(result_directory.split("/")[-1].split("instances-")[1].split("_")[0])

def get_runtime(result_directory):
    return result_directory.split("/")[-1].split("runtime-")[1].split("_")[0]

def get_featureset(result_directory):
    return result_directory.split("/")[-1].split("featureset-")[1].split("_")[0]

def get_timelimit(result_directory):
    return int(result_directory.split("/")[-1].split("timelimit-")[1])

def get_iteration(result_directory):
    if "iteration" in result_directory:
        return int(result_directory.split("/")[-1].split("iteration-")[-1].split("_")[0])
    else:
        return -1


def get_model_name(result_directory):
    if "iteration" in result_directory:
        # in this case it is a learning benchmark
        model_name, timelimit = result_directory.split("/")[-1].split("_iteration")
        return model_name + "_" + timelimit.split("_")[1]
    else:
        # in this case it is a non-learning benchmark
        return result_directory.split("/")[-1]


def check_benchmark(result_directory):
    if "monte_carlo" in result_directory:
        return True
    elif "rolling_horizon" in result_directory:
        return True
    elif "greedy" in result_directory:
        return True
    elif "lazy" in result_directory:
        return True
    elif "random" in result_directory:
        return True
    elif "oracle" in result_directory:
        return True
    else:
        return False


def check_relevant_case(result_directory, relevant_case):
    if get_strategy(result_directory) != relevant_case["strategy"]:
        return False
    if check_benchmark(result_directory):
        return True
    if get_samples(result_directory) != relevant_case["samples"]:
        return False
    if get_instances(result_directory) != relevant_case["instances"]:
        return False
    if get_runtime(result_directory) != relevant_case["runtime"]:
        return False
    if get_featureset(result_directory) != relevant_case["feature_set"]:
        return False
    if get_timelimit(result_directory) != relevant_case["timelimit"]:
        return False
    return True


def result_directory_is_relevant(result_directoy, relevant_cases):
    for relevant_case in relevant_cases:
        if check_relevant_case(result_directoy, relevant_case):
            return result_directoy
    return None

def delete_superfluous_directories(results_saver, num_instances, num_seeds):

    # get all considered seeds
    seeds = []
    solution_instances = []
    for model in results_saver.keys():
        for learning_iteration in results_saver[model].keys():
            for solution_instance in results_saver[model][learning_iteration].keys():
                solution_instances.append(solution_instance)
                seeds += list(results_saver[model][learning_iteration][solution_instance].keys())
    seeds_sorted = OrderedDict(sorted(collections.Counter(seeds).items(), key=lambda item: -1 * item[1]))
    solution_instances = list(set(solution_instances))

    for num_different_seeds in list(range(len(seeds_sorted), 0, -1)):
        instance_keeper = []
        # get x most counted seeds
        most_counted_seeds = list(seeds_sorted.keys())[:num_different_seeds]
        # keep instances that consider all most counted seeds
        for solution_instance in solution_instances:
            skip = False
            for model in results_saver.keys():
                for learning_iteration in results_saver[model].keys():
                    if any([seed not in list(results_saver[model][learning_iteration][solution_instance].keys()) for seed in most_counted_seeds]):
                        skip = True
            if not skip:
                instance_keeper.append(solution_instance)
        if len(instance_keeper) == num_instances and num_different_seeds == num_seeds:
            print("Relevant instances:" + str(instance_keeper))
            print("Relevant seeds:" + str(most_counted_seeds))
            break

    new_results_saver = {}
    for model in results_saver.keys():
        new_results_saver[model] = {}
        for learning_iteration in results_saver[model].keys():
            new_results_saver[model][learning_iteration] = {}
            for solution_instance in results_saver[model][learning_iteration].keys():
                if solution_instance in instance_keeper:
                    new_results_saver[model][learning_iteration][solution_instance] = {}
                    for seed in results_saver[model][learning_iteration][solution_instance].keys():
                        if seed in most_counted_seeds:
                            new_results_saver[model][learning_iteration][solution_instance][seed] = results_saver[model][learning_iteration][solution_instance][seed]

    return new_results_saver






def get_relevant_directories(experiment, directory):
    base_case = {"strategy": "NeuralNetwork", "samples": 50, "instances": 15, "runtime": "3600", "feature_set": "dynamicstatic", "predictor": "NeuralNetwork", "timelimit": 90}
    relevant_cases = []
    if experiment == "experiment_benchmarks":
        strategies = ["NeuralNetwork", "rolling", "monte", "oracle", "greedy", "random", "lazy"]
        for strategy in strategies:
            relevant_cases.append(set_case(base_case, "strategy", strategy))
    if experiment == "experiment_samples":
        samples = [10, 25, 50, 75, 100]
        relevant_cases.append(set_case(base_case, "strategy", "oracle"))
        for sample in samples:
            relevant_cases.append(set_case(base_case, "samples", sample))
    elif experiment == "experiment_instances":
        instances = [1, 2, 5, 10, 15, 20, 25, 30]
        relevant_cases.append(set_case(base_case, "strategy", "oracle"))
        for num_instances in instances:
            relevant_cases.append(set_case(base_case, "instances", num_instances))
    elif experiment == "experiment_runtime":
        runtimes = ["60", "120", "180", "240", "300", "900", "3600", "bestSeed"]
        relevant_cases.append(set_case(base_case, "strategy", "oracle"))
        for runtime in runtimes:
            relevant_cases.append(set_case(base_case, "runtime", runtime))
    elif experiment == "experiment_featureset":
        feature_sets = ["static", "dynamic", "dynamicstatic"]
        relevant_cases.append(set_case(base_case, "strategy", "oracle"))
        for feature_set in feature_sets:
            relevant_cases.append(set_case(base_case, "feature_set", feature_set))
    elif experiment == "experiment_predictors":
        predictors = ["NeuralNetwork", "GraphNeuralNetwork", "Linear", "GraphNeuralNetwork_sparse"]
        relevant_cases.append(set_case(base_case, "strategy", "oracle"))
        for predictor in predictors:
            relevant_cases.append(set_case(base_case, "strategy", predictor))
    elif experiment == "experiment_timelimit":
        timelimits = [30, 60, 90, 120, 180, 240]
        relevant_cases.append(set_case(base_case, "strategy", "oracle"))
        for timelimit in timelimits:
            relevant_cases.append(set_case(base_case, "timelimit", timelimit))

    directories = []
    for result_directory in os.listdir(directory):
        if result_directory_is_relevant(result_directory, relevant_cases):
            directories.append(directory + result_directory)

    return directories



def prepare_heatmap(data):

    def get_index(value):
        try:
            return int(value)
        except:
            return 0

    data_keeper = {}
    for data_point in data.keys():
        runtime, instance_size = data_point.split("_")[:-1]
        if runtime not in data_keeper.keys():
            data_keeper[runtime] = {}
        data_keeper[runtime][instance_size] = np.mean(data[data_point])
    data_keeper = OrderedDict(sorted(data_keeper.items(), key=lambda item: get_index(item[0])))
    for key in data_keeper.keys():
        data_keeper[key] = OrderedDict(sorted(data_keeper[key].items(), key=lambda item: get_index(item[0])))
    data = pd.DataFrame(data_keeper)
    return data


def plot(results_saver, experiment, sort_models, x_label, width=20, scheme="absolute", rotation=0):
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
    plt.rc('font', size=14)
    sns.set_style('whitegrid', {'axes.labelcolor': 'black'})  # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    sns.color_palette('deep')
    if x_label == "iteration":
        fig, ax = plt.subplots(figsize=(50, 25))
    else:
        fig, ax = plt.subplots(figsize=(16, 5))

    x_label_identifier = {"experiment_samples": "samples-", "experiment_instances": "instances-", "experiment_runtime": "runtime-", "experiment_featureset": "featureset-", "experiment_timelimit": "timelimit-"}
    x_labels_dict = {"lazy": "lazy", "random": "random", "greedy": "greedy", "rolling": "rolling-horizon",
                     "monte": "monte-carlo", "NeuralNetwork": ("ML-CO" if experiment == "experiment_benchmarks" else "Neural Network"), "oracle": "anticipative \\\\ lower bound",
                     "GraphNeuralNetwork": "Graph Neural \\\\ Network", "GraphNeuralNetwork_sparse": "Graph Neural \\\\ Network (sparse)", "NeuralNetworkInd": "Ind. NNs"}
    extra_axis_parameters = {"variance": ["xmajorticks=true", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}", "ytick style={draw=none}", "xticklabel style={align=center}", "yticklabel style={xshift=-4pt}"]}
    color_palette = {"lazy": sns.color_palette()[6], "random": sns.color_palette()[5], "greedy": sns.color_palette()[4], "rolling-horizon": sns.color_palette()[3], "monte-carlo": sns.color_palette()[2], "ML-CO": sns.color_palette()[1], "anticipative \\\\ lower bound": sns.color_palette()[0]}

    def get_axis_parameter(scheme):
        if scheme in extra_axis_parameters.keys():
            return extra_axis_parameters[scheme]
        else:
            return ["xmajorticks=true", "ymajorticks=true", "ylabel style={align=center}", "scaled y ticks=true", "xtick style={draw=none}", "ytick style={draw=none}", "xticklabel style={align=center}"]


    def get_color(x_tick):
        if x_tick in color_palette.keys():
            return color_palette[x_tick]
        else:
            return sns.color_palette()[1]


    def get_x_label(model):
        if model in x_labels_dict.keys():
            return x_labels_dict[model]
        else:
            return model


    position_end = 0
    if scheme == "relative_best":
        results_saver = mean_seeds(results_saver)
        updated_results_saver = get_relative_best_values(results_saver)
        metric = "Relative distance to best solution"
        updated_results_saver = sort(updated_results_saver)
        relevant_models = updated_results_saver.keys()
    elif scheme == "relative_greedy":
        results_saver = mean_seeds(results_saver)
        updated_results_saver = get_relative_greedy_values(results_saver)
        metric = "Relative distance to greedy solution"
        updated_results_saver = sort(updated_results_saver)
        relevant_models = updated_results_saver.keys()
    elif scheme == "relative_monte_carlo":
        results_saver = mean_seeds(results_saver)
        updated_results_saver = get_relative_monte_carlo_values(results_saver)
        metric = "Relative distance to monte carlo solution"
        updated_results_saver = sort(updated_results_saver)
        relevant_models = updated_results_saver.keys()
    elif scheme == "model_per_instance_relative_oracle":
        results_saver = mean_seeds(results_saver)
        updated_results_saver = get_model_per_instance_values(results_saver)
        updated_results_saver = delete_non_consistend_instances(updated_results_saver)
        updated_results_saver = get_relative_oracle_values(updated_results_saver)
        metric = "Relative distance to best solution"
        updated_results_saver = sort(updated_results_saver)
        relevant_models = [mode for mode in updated_results_saver.keys() if mode != "oracle"]
    elif scheme == "variance":
        updated_results_saver = variance_seeds(results_saver)
        updated_results_saver = get_absolute_values(updated_results_saver)
        metric = "Relative distance to best solution"
        updated_results_saver = sort(updated_results_saver)
        relevant_models = updated_results_saver.keys()
    elif scheme == "absolute":
        results_saver = mean_seeds(results_saver)
        updated_results_saver = get_absolute_values(results_saver)
        updated_results_saver = clean_from_missing_values(updated_results_saver)
        metric = "Mean cost"
        updated_results_saver = sort(updated_results_saver)
        relevant_models = updated_results_saver.keys()
    elif scheme == "relative_oracle":
        results_saver = mean_seeds(results_saver)
        updated_results_saver = get_relative_oracle_values(results_saver)
        metric = "Relative distance to ant. lb."
        updated_results_saver = sort(updated_results_saver)
        relevant_models = [mode for mode in updated_results_saver.keys() if mode != "oracle_timelimit-3600"]
    else:
        raise Exception("Wrong scheme defined.")

    def get_sort_int(value):
        try:
            return int(value)
        except:
            return 10000


    if sort_models:
        relevant_models = sorted(relevant_models, key=lambda item: get_sort_int(item.split(x_label_identifier[experiment])[-1].split("_")[0]))

    data_list = {}
    colors = []
    for model in relevant_models:
        position_end += len(updated_results_saver[model])
        if x_label == "iteration":
            x_labels = [get_x_label(model) + "/" + str(iteration) for iteration in list(updated_results_saver[model].keys())]
        elif x_label == "strategies":
            x_labels = [get_x_label(model.split("_")[0])]
        elif x_label == "predictors":
            x_labels = [get_x_label(model.split("_samples")[0])]
        else:
            x_labels = [get_x_label(model.split(x_label_identifier[experiment])[-1].split("_")[0])]
        for x_tick, values in zip(x_labels, list(updated_results_saver[model].values())):
            data_list[x_tick] = np.array(values)
            colors.append(get_color(x_tick))
        print("{}: {}={}".format(metric, model, np.mean(list(updated_results_saver[model].values()))))
        plt.xticks(rotation=rotation)
    if experiment == "experiment-8_":
        heatmap = prepare_heatmap(data_list)
        sns.heatmap(heatmap)
    else:
        sns.boxplot(data=pd.DataFrame(data_list), palette=colors)
    if scheme == "relative_oracle":
        plt.ylabel("Relative distance \\\\ to ant. lb. [\%]")
    elif scheme == "relative_best":
        plt.ylabel("Relative distance to\\\\best found solution [\%]")
    elif scheme == "relative_greedy":
        plt.ylabel("Relative distance \\\\ to greedy solution [\%]")
    elif scheme == "relative_monte_carlo":
        plt.ylabel("Relative distance \\\\ to monte carlo solution [\%]")
    elif scheme == "variance":
        plt.ylabel("Variance of obj. value across \\\\ instance seeds")
    elif scheme == "absolute":
        plt.ylabel("Sample size training instances")
    plt.tight_layout()
    tikzplotlib.save("./figures/evaluate_{}_{}.tex".format(experiment, scheme), axis_width=f"{width}cm", axis_height="5cm",
                     extra_axis_parameters=get_axis_parameter(scheme))
    plt.show()


def get_best_learning_iteration(results_saver):
    best_iteration_saver = OrderedDict()
    results_saver = mean_seeds(results_saver)
    for mode in results_saver.keys():
        best_value = -100000000000
        best_iteration = -1
        for learning_iteration in results_saver[mode].keys():
            mean_value = sum(list(results_saver[mode][learning_iteration].values())) / len(list(results_saver[mode][learning_iteration].values()))
            if mean_value > best_value:
                best_value = mean_value
                best_iteration = learning_iteration
        best_iteration_saver[mode] = best_iteration
        print(f"Best learning iteration mode {mode} : {best_iteration}")
    return best_iteration_saver

def set_best_learning_iteration(saver, best_learning_iteration):
    for mode in saver.keys():
        saver[mode] = saver[mode][best_learning_iteration[mode]]
    return saver


def mean_seeds(results_saver):
    for key in results_saver.keys():
        for learning_iteration in results_saver[key].keys():
            for instance in results_saver[key][learning_iteration]:
                results_saver[key][learning_iteration][instance] = np.mean(list(results_saver[key][learning_iteration][instance].values()))
    return results_saver

def variance_seeds(results_saver):
    for key in results_saver.keys():
        for learning_iteration in results_saver[key].keys():
            for instance in results_saver[key][learning_iteration]:
                results_saver[key][learning_iteration][instance] = np.var(list(results_saver[key][learning_iteration][instance].values()))
    return results_saver



def show_learning_evolution(directory, experiment):
    not_relevant_instances = []
    relevant_directories = get_relevant_directories(experiment, directory)
    results_saver = fill_results_saver(relevant_directories=relevant_directories, experiment=experiment, scheme="absolute", not_relevant_instances=not_relevant_instances)
    best_learning_iteration = get_best_learning_iteration(copy.deepcopy(results_saver))
    plot(results_saver, experiment, sort_models=False, x_label="iteration", rotation=90)


def show_results(directory, experiment, scheme, x_label, sort_models=True, width="5"):
    print(f"Experiment: {experiment}")
    relevant_directories = get_relevant_directories(experiment, directory)
    relevant_directories_benchmarks = get_relevant_directories("experiment_benchmarks", directory)
    not_relevant_instances = get_not_relevant_instances(list(set(relevant_directories+relevant_directories_benchmarks)))
    results_saver = fill_results_saver(relevant_directories, experiment, scheme, not_relevant_instances)
    results_saver = delete_superfluous_directories(results_saver, num_instances=25, num_seeds=20)
    plot(results_saver, experiment, sort_models, x_label, width, scheme)




if __name__ == '__main__':
    #show_learning_evolution(directory="../results/results_validation_final/", experiment="experiment_benchmarks")
    #show_learning_evolution(directory="../results/results_validation_final/", experiment="experiment_samples")
    #show_learning_evolution(directory="../results/results_validation_final/", experiment="experiment_instances")
    #show_learning_evolution(directory="../results/results_validation_final/", experiment="experiment_runtime")
    #show_learning_evolution(directory="../results/results_validation_final/", experiment="experiment_featureset")
    #show_learning_evolution(directory="../results/results_validation_final/", experiment="experiment_predictors")

    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_benchmarks", scheme="absolute", x_label="strategies", sort_models=False, width="16")
    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_benchmarks", scheme="relative_oracle", x_label="strategies", sort_models=False, width="16")
    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_benchmarks", scheme="variance", x_label="strategies", sort_models=False, width="16")
    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_samples", scheme="relative_oracle", x_label="size_instances", width="10")
    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_instances", scheme="relative_oracle", x_label="num_instances")
    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_runtime", scheme="relative_oracle", x_label="oracle_solution")
    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_featureset", scheme="relative_oracle", x_label="feature_set", sort_models=False)
    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_predictors", scheme="relative_oracle", x_label="predictors", sort_models=False, width="11")
    show_results(directory="../results/results_test_seeds_final/", experiment="experiment_timelimit", scheme="relative_oracle", x_label="timelimit", sort_models=False, width="11")