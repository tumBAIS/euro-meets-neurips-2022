import pandas as pd
import numpy as np
from training.src import util


class FeatureComputer():
    def __init__(self, args, observation, instance, static_info):
        self.args = args
        self.EPOCH_DURATION = 3600
        self.MARGIN_DISPATCH = 3600
        self.instance = instance  # this gives the instance from tools.read_vrplib(args.instance)
        self.travel_time = instance["duration_matrix"]
        self.static_info = static_info
        self.observation = observation
        self.current_epoch = observation["current_epoch"]
        self.current_time = observation["current_time"]
        self.planning_starttime = observation["planning_starttime"]
        self.epoch_instance_numpy = observation["epoch_instance"]
        self.duration_matrix = self.epoch_instance_numpy["duration_matrix"]
        self.epoch_instance = pd.DataFrame({"is_depot": self.epoch_instance_numpy["is_depot"],
                                                   "customer_idx": self.epoch_instance_numpy["customer_idx"],
                                                   "request_idx": self.epoch_instance_numpy["request_idx"],
                                                   "coords_x": self.epoch_instance_numpy["coords"][:, 0],
                                                   "coords_y": self.epoch_instance_numpy["coords"][:, 1],
                                                   "demands": self.epoch_instance_numpy["demands"],
                                                   "time_windows_start": self.epoch_instance_numpy["time_windows"][:, 0],
                                                   "time_windows_end": self.epoch_instance_numpy["time_windows"][:, 1],
                                                   "service_times": self.epoch_instance_numpy["service_times"],
                                                   "must_dispatch": self.epoch_instance_numpy["must_dispatch"],
                                                   "current_epoch": self.current_epoch})
        self.list_feature_names = []
        self.list_edge_feature_names = []
        self.get_nodes()
        self.get_edges()
        self.get_graph_edges()
        self.edge_features = pd.DataFrame(self.graph_edges, columns=["starting", "arriving"])

    def append_feature_to_feature_list(self, feature_name):
        self.list_feature_names.append(feature_name)

    def append_feature_to_edge_feature_list(self, feature_name):
        self.list_edge_feature_names.append(feature_name)

    # FEATURE number requests
    def get_feature_mustDispatch(self):
        self.epoch_instance["must_dispatch_feature"] = self.epoch_instance["must_dispatch"].astype(int)
        self.append_feature_to_feature_list("must_dispatch_feature")

    # FEATURE time depot -> request
    def get_feature_timeDepotRequest(self):
        timeDepotRequest = self.duration_matrix[0, :]
        self.epoch_instance["timeDepotLocation"] = timeDepotRequest
        self.append_feature_to_feature_list("timeDepotLocation")

    # FEATURE location coordinates
    def get_feature_coords(self):
        self.epoch_instance["feature_coords_x"] = self.epoch_instance_numpy["coords"][:, 0]
        self.epoch_instance["feature_coords_y"] = self.epoch_instance_numpy["coords"][:, 1]
        self.append_feature_to_feature_list("feature_coords_x")
        self.append_feature_to_feature_list("feature_coords_y")

    # FEATURE demands
    def get_feature_demands(self):
        self.epoch_instance["feature_demands"] = self.epoch_instance_numpy["demands"]
        self.append_feature_to_feature_list("feature_demands")

    # FEATURE service times
    def get_feature_serviceTimes(self):
        self.epoch_instance["feature_serviceTime"] = self.epoch_instance_numpy["service_times"]
        self.append_feature_to_feature_list("feature_serviceTime")

    # FEATURE time windows
    def get_feature_timeWindows(self):
        self.epoch_instance["feature_timeWindowsStart"] = self.epoch_instance_numpy["time_windows"][:, 0]
        self.epoch_instance["feature_timeWindowsEnd"] = self.epoch_instance_numpy["time_windows"][:, 1]
        self.append_feature_to_feature_list("feature_timeWindowsStart")
        self.append_feature_to_feature_list("feature_timeWindowsEnd")

    # FEATURE timeDepotLocation / (timeWindowEnd - serviceTime)
    def get_feature_ratio_timeDepotLocation_timeWindowEndMinusServiceTime(self):
        timeDepotRequest = self.duration_matrix[0, :]
        windowEndMinusServiceTime = (self.epoch_instance["time_windows_end"] - self.epoch_instance["service_times"]).replace(0, 1)
        relative_timeDepotLocation_windowEndMinusServiceTime = timeDepotRequest / windowEndMinusServiceTime
        self.epoch_instance["relative_timeDepotLocation_windowEnd"] = relative_timeDepotLocation_windowEndMinusServiceTime
        self.append_feature_to_feature_list("relative_timeDepotLocation_windowEnd")

    # FEATURE quantile features : time to other requests
    def get_feature_quantileTimeToRequests(self,):
        quantiles = [0.01, .05, .10, .5]
        quantiles_time_toRequests = np.quantile(self.travel_time[self.epoch_instance["customer_idx"], :],
                                                q=quantiles, axis=1)
        for q_idx, q in enumerate(quantiles):
            self.epoch_instance["q_{}_time".format(q)] = quantiles_time_toRequests[q_idx, :]
            self.append_feature_to_feature_list("q_{}_time".format(q))

    # FEATURE normalized time windows
    def get_feature_normalizedTimeWindows(self):
        end_time = (self.EPOCH_DURATION * self.static_info["end_epoch"]) + self.MARGIN_DISPATCH
        remaining_time = end_time - self.planning_starttime
        self.epoch_instance['relative_time_window_start'] = self.epoch_instance['time_windows_start'] / remaining_time
        self.append_feature_to_feature_list("relative_time_window_start")
        self.epoch_instance['relative_time_window_end'] = self.epoch_instance['time_windows_end'] / remaining_time
        self.append_feature_to_feature_list("relative_time_window_end",)

    # FEATURE quantile features : remaining time windows
    def get_feature_quantileTimeRemainingTimeWindows(self,):
        # consider time windows that we can reach after the request
        current_time_windows_after = self.instance["time_windows"][:, 0][self.epoch_instance_numpy["customer_idx"]]
        relevant_time_windows_after = self.instance["time_windows"][:, 1][
            self.instance["time_windows"][:, 1] > self.planning_starttime]
        average_travel_time_after = np.mean(self.travel_time[self.epoch_instance["customer_idx"], :], axis=1)
        relevant_time_windows_after = np.rot90(
            np.repeat(relevant_time_windows_after[:, np.newaxis], len(self.epoch_instance), axis=1))[:, 1:]
        relevant_time_windows_after = np.clip(np.subtract(relevant_time_windows_after,
                                                          np.repeat(current_time_windows_after[:, np.newaxis],
                                                                    relevant_time_windows_after.shape[1], axis=1)),
                                              a_min=0, a_max=None)  # subtract current time window
        relevant_time_windows_after = np.clip(np.subtract(relevant_time_windows_after, np.repeat(
            self.epoch_instance_numpy["service_times"][:, np.newaxis], relevant_time_windows_after.shape[1], axis=1)),
                                              a_min=0, a_max=None)  # subtract service time
        relevant_time_windows_after = np.clip(np.subtract(relevant_time_windows_after,
                                                          np.repeat(average_travel_time_after[:, np.newaxis],
                                                                    relevant_time_windows_after.shape[1], axis=1)),
                                              a_min=0, a_max=None)  # subtract average travel time
        # consider time windows that we can reach before the request
        current_time_windows_before = self.instance["time_windows"][:, 1][self.epoch_instance_numpy["customer_idx"]]
        relevant_time_windows_before = self.instance["time_windows"][:, 0][
            self.instance["time_windows"][:, 1] > self.planning_starttime]
        average_service_time = np.mean(self.instance["service_times"][1:])
        relevant_time_windows_before = np.clip(relevant_time_windows_before, a_min=self.planning_starttime, a_max=None)
        average_travel_time_before = np.mean(self.travel_time[:, self.epoch_instance["customer_idx"]], axis=0)
        current_time_windows_before = np.repeat(current_time_windows_before[:, np.newaxis],
                                                len(relevant_time_windows_before) - 1,
                                                axis=1)  # we subtract 1 that the shapes are consistent as we ignore the depot time window in the next line
        current_time_windows_before = np.clip(np.subtract(current_time_windows_before,
                                                          np.repeat(relevant_time_windows_before[np.newaxis, :],
                                                                    current_time_windows_before.shape[0], axis=0)[:,
                                                          1:]), a_min=0, a_max=None)  # subtract current time window
        current_time_windows_before = np.clip(np.subtract(current_time_windows_before, average_service_time), a_min=0,
                                              a_max=None)  # subtract service time
        current_time_windows_before = np.clip(np.subtract(current_time_windows_before,
                                                          np.repeat(average_travel_time_before[:, np.newaxis],
                                                                    current_time_windows_before.shape[1], axis=1)),
                                              a_min=0, a_max=None)  # subtract average travel time
        relevant_time_windows = np.maximum(current_time_windows_before, relevant_time_windows_after)
        quantiles = [0, 0.01, .05, .10, .5]
        quantiles_timeWindows = np.quantile(relevant_time_windows, q=quantiles, axis=1)
        for q_idx, q in enumerate(quantiles):
            self.epoch_instance["q_{}_timeWindows".format(q)] = quantiles_timeWindows[q_idx, :]
            self.append_feature_to_feature_list("q_{}_timeWindows".format(q))

    def get_feature_distance(self):
        self.edge_features["distance"] = np.expand_dims(self.duration_matrix.flatten(order='C')[self.adjacency_matrix_graph.ravel(order='C').astype(bool)], 1)
        self.append_feature_to_edge_feature_list("distance")

    def get_target(self, solution):
        self.epoch_instance["target"] = self.epoch_instance.apply(lambda node_features: True if node_features["request_idx"] in solution else False, axis=1)

    def get_nodes(self):
        self.nodes = np.array(self.epoch_instance["request_idx"])

    def get_edges(self):
        self.adjacency_matrix = np.ones((len(self.nodes), len(self.nodes))).astype(int)
        np.fill_diagonal(self.adjacency_matrix, 0)
        XX, YY = np.meshgrid(self.nodes, self.nodes)
        indices = np.array([YY.ravel(order='C'), XX.ravel(order='C')]).T
        self.edges = indices.tolist()

    def get_graph_edges(self):
        self.adjacency_matrix_graph = np.ones((len(self.nodes), len(self.nodes))).astype(int)
        np.fill_diagonal(self.adjacency_matrix_graph, 0)
        if self.args.predictor == "GraphNeuralNetwork_sparse":
            starting = np.repeat(self.epoch_instance_numpy["time_windows"][:, 0][:, np.newaxis], len(self.epoch_instance_numpy["time_windows"]), axis=1)
            arriving = np.repeat(self.epoch_instance_numpy["time_windows"][:, 1][np.newaxis, :], len(self.epoch_instance_numpy["time_windows"]), axis=0)
            feasible = (starting + self.duration_matrix) < arriving
            self.adjacency_matrix_graph = np.logical_and(self.adjacency_matrix_graph, feasible)
        XX, YY = np.meshgrid(np.arange(self.adjacency_matrix_graph.shape[1]), np.arange(self.adjacency_matrix_graph.shape[0]))
        indices = np.array([YY.ravel(order='C'), XX.ravel(order='C')]).T
        self.graph_edges = indices[self.adjacency_matrix_graph.ravel(order='C').astype(bool)].tolist()

    def get_features(self):
        return self.epoch_instance[self.list_feature_names]

    def get_edge_features(self):
        return self.edge_features[self.list_edge_feature_names]


def create_features(args, training_instances):
    training_set = []
    for training_instance in training_instances:
        for epoch_observation, epoch_solution in zip(training_instance["epoch_observations"], training_instance["epoch_solutions"]):
            training_set.append(format_features(args, observation=epoch_observation,
                                          instance=training_instance["instance"],
                                          static_info=training_instance["static_info"],
                                          solution=epoch_solution))
    return training_set


def format_features(args, observation, instance, static_info, solution=None):
    featureComputer = run_feature_computer(args, observation, instance, static_info, solution)
    feature_dict = {"epoch_instance": featureComputer.epoch_instance,
                         "features": featureComputer.get_features(),
                         "edge_features": featureComputer.get_edge_features(),
                         "edges": featureComputer.edges,
                         "graph_edges": featureComputer.graph_edges,
                         "nodes": featureComputer.nodes,
                         "epoch_solution": solution,
                         "duration_matrix": featureComputer.duration_matrix,
                         "epoch_instance_numpy": featureComputer.epoch_instance_numpy}
    return feature_dict


def run_feature_computer(args, observation, instance, static_info, solution=None):
    fc = FeatureComputer(args, observation, instance, static_info)

    # node features
    if "dynamic" in args.feature_set:
        fc.get_feature_coords()
        fc.get_feature_demands()
        fc.get_feature_serviceTimes()
        fc.get_feature_timeWindows()
        fc.get_feature_timeDepotRequest()
        fc.get_feature_ratio_timeDepotLocation_timeWindowEndMinusServiceTime()
        fc.get_feature_normalizedTimeWindows()
        fc.get_feature_mustDispatch()
    if "static" in args.feature_set:
        fc.get_feature_quantileTimeToRequests()
        fc.get_feature_quantileTimeRemainingTimeWindows()

    # edge features
    fc.get_feature_distance()


    if isinstance(solution, list):
        fc.get_target([node for node_list in solution for node in node_list])

    return fc
