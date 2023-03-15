import numpy as np
from collections import Counter

class Sample_Generator:
    def __init__(self, args, static_info):
        self.instance = static_info["dynamic_context"]
        self.start_epoch = static_info["start_epoch"]
        self.end_epoch = static_info["end_epoch"]
        self.EPOCH_DURATION = 3600
        self.MAX_REQUESTS_PER_EPOCH = 100
        self.rng = np.random.default_rng(args.sample_seed)

    def sample(self, observation):
        current_epoch = observation["current_epoch"]
        planning_starttime = observation["planning_starttime"]
        epoch_instance = observation["epoch_instance"]
        MARGIN_DISPATCH = observation["planning_starttime"] - (current_epoch * self.EPOCH_DURATION)
        start_requests_idx = np.max(observation["epoch_instance"]["request_idx"]) + 1
        num_requests = len(observation["epoch_instance"]["coords"])
        duration_matrix = self.instance['duration_matrix']

        # Sample uniformly
        num_customers = len(self.instance['coords']) - 1  # Exclude depot

        # Sample data uniformly from customers (1 to num_customers)
        def sample_from_customers(k=self.MAX_REQUESTS_PER_EPOCH):
            return self.rng.integers(num_customers, size=k) + 1

        sampled_offline_observation = epoch_instance  # collects sampled observations for each future simulated epoch
        sampled_offline_observation["release_times"] = np.full(num_requests, planning_starttime)
        sampled_offline_observation["release_times"][self.instance['is_depot'][epoch_instance["customer_idx"]]] = 0
        for simulated_epoch_idx in list(range(current_epoch + 1, self.end_epoch + 1)):
            current_time_simulated_epoch = self.EPOCH_DURATION * simulated_epoch_idx
            planning_starttime_simulated_epoch = current_time_simulated_epoch + MARGIN_DISPATCH

            cust_idx = sample_from_customers()
            timewi_idx = sample_from_customers()
            demand_idx = sample_from_customers()
            service_t_idx = sample_from_customers()

            new_request_timewi = self.instance['time_windows'][timewi_idx]

            # Filter data that can no longer be delivered
            # Time + margin for dispatch + drive time from depot should not exceed latest arrival
            is_feasible = planning_starttime_simulated_epoch + duration_matrix[0, cust_idx] <= new_request_timewi[:, 1]

            if is_feasible.any():
                num_new_requests = is_feasible.sum()
                request_id = np.arange(num_new_requests) + start_requests_idx
                request_customer_index = cust_idx[is_feasible]
                request_timewi = new_request_timewi[is_feasible]
                request_service_t = self.instance['service_times'][service_t_idx[is_feasible]]
                request_demand = self.instance['demands'][demand_idx[is_feasible]]
                release_times = np.full(num_new_requests, planning_starttime_simulated_epoch)

            start_requests_idx = start_requests_idx + num_new_requests

            # Renormalize time to start at planning_starttime, and clip time windows in the past (so depot will start at 0)
            time_windows = np.clip(request_timewi - planning_starttime, a_min=0, a_max=None)
            # concatenate sampled observation per epoch to sampled offline observation
            sampled_offline_observation = {
                'is_depot': np.concatenate((sampled_offline_observation["is_depot"], np.array(self.instance['is_depot'][request_customer_index]))),
                'customer_idx': np.concatenate((sampled_offline_observation["customer_idx"], request_customer_index)),
                'request_idx': np.concatenate((sampled_offline_observation["request_idx"], request_id)),
                'coords': np.concatenate((sampled_offline_observation["coords"], np.array(self.instance['coords'][request_customer_index]))),
                'demands': np.concatenate((sampled_offline_observation["demands"], request_demand)),
                'capacity': self.instance['capacity'],
                'time_windows': np.concatenate((sampled_offline_observation["time_windows"], time_windows)),
                'service_times': np.concatenate((sampled_offline_observation["service_times"], request_service_t)),
                'release_times': np.concatenate((sampled_offline_observation["release_times"], release_times))
            }

        sampled_offline_observation["duration_matrix"] = self.instance['duration_matrix'][np.ix_(sampled_offline_observation["customer_idx"], sampled_offline_observation["customer_idx"])]
        # Renormalize release_times to start at planning_starttime
        sampled_offline_observation["release_times"][~sampled_offline_observation["is_depot"]] = sampled_offline_observation["release_times"][~sampled_offline_observation["is_depot"]] - planning_starttime
        sampled_offline_observation["release_times"][~sampled_offline_observation["is_depot"]] = sampled_offline_observation["release_times"][~sampled_offline_observation["is_depot"]] + MARGIN_DISPATCH
        sampled_offline_observation["time_windows"] = sampled_offline_observation["time_windows"] + MARGIN_DISPATCH
        sampled_offline_observation["time_windows"][sampled_offline_observation["is_depot"], 0] = 0

        return sampled_offline_observation


    def build_observation_from_requestidxs(self, request_idxs, observation):

        bool_in_new_observation = [True if request_idx in request_idxs else False for request_idx in observation["epoch_instance"]["request_idx"]]
        bool_in_new_observation[0] = True
        customer_idx = observation["epoch_instance"]["customer_idx"][bool_in_new_observation]
        planning_starttime = observation["planning_starttime"]
        current_epoch = observation["current_epoch"]
        current_time = observation["current_time"]

        epoch_instance = {
            'is_depot': self.instance['is_depot'][customer_idx],
            'customer_idx': customer_idx,
            'request_idx': observation["epoch_instance"]["request_idx"][bool_in_new_observation],
            'coords': self.instance['coords'][customer_idx],
            'demands': observation["epoch_instance"]["demands"][bool_in_new_observation],
            'capacity': self.instance['capacity'],
            'time_windows': observation["epoch_instance"]["time_windows"][bool_in_new_observation],
            'service_times': observation["epoch_instance"]["service_times"][bool_in_new_observation],
            'duration_matrix': self.instance['duration_matrix'][np.ix_(customer_idx, customer_idx)],
            'must_dispatch': np.ones(shape=len(request_idxs)),
        }
        return {
            'current_epoch': current_epoch,
            'current_time': current_time,
            'planning_starttime': planning_starttime,
            'epoch_instance': epoch_instance
        }

    def strategy_to_choose_route(self, args, observation, sample_solutions):
        epoch_instance = observation["epoch_instance"]
        request_idx_set = set(epoch_instance['request_idx'])

        ### Here we have to answer the question which request we want to dispatch
        if args.strategy == "rolling_horizon":
            ## 1. -> dispatch routes that only consist of requests from observation + dispatch routest that contain must_dispatch requests
            epoch_solution = [route.tolist() for route in sample_solutions[0]]
            # 1.1 -> dispatch routes that only consist of requests from observation
            epoch_solution_only_observed_requests = [route for route in epoch_solution if len(request_idx_set.intersection(route)) == len(route)]
            remaining_epoch_solution = [route for route in epoch_solution if route not in epoch_solution_only_observed_requests]
            # 1.2 -> dispatch routes that contain must_dispatch requests
            epoch_solution_must_dispatch_with_sampled = [route for route in remaining_epoch_solution if
                                                         any(request in epoch_instance["request_idx"][epoch_instance["must_dispatch"]] for request in route)] # get routes that we have to dispatch
            if len(epoch_solution_must_dispatch_with_sampled) > 0:
                print("epoch_solution_must_dispatch_with_sampled > 0 -> should not happen as all must-dispatch should be in route already!")
            epoch_solution_must_dispatch = [[request for request in route if request in epoch_instance["request_idx"]] for route in epoch_solution_must_dispatch_with_sampled]  # delete sampled requests
            epoch_solution = epoch_solution_only_observed_requests + epoch_solution_must_dispatch
            request_idxs = [request_idx for route in epoch_solution for request_idx in route]
        elif args.strategy == "monte_carlo":
            # dispatch all requests that are in more than 50% of routes
            routes = [list(route) for sample_solution in sample_solutions for route in sample_solution]
            routes_only_observed_requests = [route for route in routes if len(request_idx_set.intersection(route)) == len(route)]
            request_idxs = [request_idx for route in routes_only_observed_requests for request_idx in route]
            request_idxs_cnt = Counter(request_idxs)
            request_idxs = [request_idx for request_idx, count in request_idxs_cnt.most_common() if count > int(args.monte_carlo_sampling_rounds/2)]
            # only incorporate real requests
            request_idxs = [request_idx for request_idx in request_idxs if request_idx in epoch_instance["request_idx"]]
        else:
            raise Exception("Wrong sample strategy defined")

        observation = self.build_observation_from_requestidxs(request_idxs, observation)
        return observation