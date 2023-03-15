import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, TextIO
from datetime import timedelta
import numpy as np
import tools
from environment import VRPEnvironment
from baselines.strategies import STRATEGIES
from tools import solve_static_vrptw, solve_pchgs
from src import config, util
import dataclass_factory
from SampleGenerator import Sample_Generator
from multiprocessing import Pool
import multiprocessing as mp
import os
from functools import partial


def validate(oracle_solution, env: VRPEnvironment) -> Tuple[Optional[float], List[List[int]]]:
    total_reward = 0
    epoch_solutions = []
    observation, static_info = env.reset()
    while True:
        epoch_instance = observation['epoch_instance']

        request_idx = set(epoch_instance['request_idx'])
        epoch_solution = [route for route in oracle_solution if len(request_idx.intersection(route)) == len(route)]
        cost = tools.validate_dynamic_epoch_solution(epoch_instance, epoch_solution)

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)
        assert cost is None or reward == -cost, "Reward should be negative cost of solution"
        if info['error']:
            return None, []

        epoch_solutions.append((epoch_solution, reward))
        total_reward += reward

        if done:
            break

    return total_reward, epoch_solutions


@dataclass
class Route:
    customers: List[int]
    epoch: Optional[int]


@dataclass
class Solution:
    routes: List[Route]
    cost: float
    found_after_ms: Optional[int]

    @property
    def raw_routes(self) -> List[List[int]]:
        return [x.customers for x in self.routes]


def create_solution_from_oracle(routes: List[List[int]], cost: float, found_at_ns: int,
                                env: VRPEnvironment) -> Solution:
    total_reward, epoch_solutions = validate(routes, env)
    return Solution(
        [Route(route, epoch=epoch + 1) for epoch, epoch_sol in enumerate(epoch_solutions) for route in epoch_sol[0]],
        cost=cost, found_after_ms=found_at_ns // 1000)


def create_solution_from_dynamic(epoch_solution: List[List[int]], cost: float, epoch: int,
                                 found_at_ns: int) -> Solution:
    return Solution([Route(route, epoch=epoch + 1) for route in epoch_solution], cost=cost,
                    found_after_ms=found_at_ns // 1000)


def run_oracle(args, env: VRPEnvironment, executable: str) -> List[Solution]:
    if not Path(executable).exists():
        raise ValueError(f"HGS Executable not found: {executable}")
    # Oracle strategy which looks ahead, this is NOT a feasible strategy but gives a 'bound' on the performance
    # Bound written with quotes because the solution is not optimal so a better solution may exist
    # This oracle can also be used as supervision for training a model to select which requests to dispatch

    # First get hindsight problem (each request will have a release time)
    done = False
    observation, info = env.reset()
    epoch_tlim = args.time_limit
    while not done:
        # Dummy solution: 1 route per request
        epoch_solution = [[request_idx] for request_idx in observation['epoch_instance']['request_idx'][1:]]
        observation, reward, done, info = env.step(epoch_solution)
    hindsight_problem = env.get_hindsight_problem()

    # Get static solutions
    solutions = []
    for sol in solve_static_vrptw(args, hindsight_problem, executable, time_limit=epoch_tlim):
        routes, cost, ns_offset = sol
        solve_ms = ns_offset // 1000.0
        print(f"[{timedelta(microseconds=solve_ms)}] Found solution ({sol[1]}) {routes}")
        solutions.append(sol)

    # Build solutions
    solutions = [create_solution_from_oracle(routes, cost, ns_offset // 1000, env) for routes, cost, ns_offset in
                 solutions]

    # Get best sol
    oracle_solution = min(solutions, key=lambda x: x.cost)
    oracle_cost = tools.validate_static_solution(hindsight_problem, oracle_solution.raw_routes)

    total_reward = oracle_solution.cost
    if total_reward is None or total_reward != oracle_cost:
        raise ValueError("Failed to validate oracle solution!")

    return solutions


def run_strategy(args, env: VRPEnvironment, executable: str, solver_seed: int, strategy,
                 instance) -> Tuple[Solution, List[Solution]]:
    rng = np.random.default_rng(solver_seed)


    if args.strategy in ["NeuralNetwork", "Linear", "GraphNeuralNetwork", "GraphNeuralNetwork_sparse"]:
        model_SL = util.load_SL_model(args=args, directory=args.model_name)

    total_reward = 0
    solutions = []
    best_solution = []
    observation, static_info = env.reset()
    if args.strategy in ["rolling_horizon", "monte_carlo"]:
        sample_generator = Sample_Generator(args, static_info)
    epoch_tlim = args.time_limit
    epoch_idx = 0
    while True:
        epoch_evaluation_start_time = time.perf_counter()
        epoch_instance = observation['epoch_instance']
        current_epoch = observation['current_epoch']
        print("Current epoch: {}. Requests: {}. Must dispatch: {}. Instance: {}".format(current_epoch, len(epoch_instance['request_idx']) - 1, epoch_instance['must_dispatch'].sum(), args.instance))

        # Select the requests to dispatch using the strategy
        if args.strategy in ["rolling_horizon", "monte_carlo"]:
            sample_solutions = []
            # Sample future observations and find routes
            for _ in list(range(args.monte_carlo_sampling_rounds)):
                epoch_instance_dispatch = sample_generator.sample(observation)
                epoch_solutions = list(solve_static_vrptw(args, epoch_instance_dispatch, executable,
                                                          time_limit=int((epoch_tlim - 90) / args.monte_carlo_sampling_rounds),
                                                          seed=solver_seed))
                epoch_solution, cost, found_at_ns = epoch_solutions[-1]
                epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution if len(route) > 0]
                sample_solutions.append(epoch_solution)
            # choose requests to dispatch from sampled solution routes
            observation = sample_generator.strategy_to_choose_route(args, observation=observation, sample_solutions=sample_solutions)
            epoch_instance_dispatch = strategy(observation["epoch_instance"], rng)
        elif args.strategy in ["NeuralNetwork", "Linear", "GraphNeuralNetwork", "GraphNeuralNetwork_sparse"]:
            epoch_instance_dispatch = strategy(args, observation=observation, model=model_SL, instance=instance, static_info=static_info)
        else:
            epoch_instance_dispatch = strategy(epoch_instance, rng)

        time_passed_instance_start_of_evaluation = time.perf_counter() - epoch_evaluation_start_time
        # Run HGS with time limit and get last solution (= best solution found)
        # Note we use the same solver_seed in each epoch: this is sufficient as for the static problem
        # we will exactly use the solver_seed whereas in the dynamic problem randomness is in the instance
        print("remaining_time before hgs: {}".format(str(epoch_tlim - time_passed_instance_start_of_evaluation)))
        if args.strategy in ["NeuralNetwork", "Linear", "GraphNeuralNetwork", "GraphNeuralNetwork_sparse"]:
            solution_pchgs = solve_pchgs(args=args, instance=epoch_instance_dispatch["pchgs_instance"], executable=executable,
                                         time_limit=int(epoch_tlim - 2 - time_passed_instance_start_of_evaluation), seed=solver_seed)
            epoch_solution = [route["requests"] for route in solution_pchgs["routes"]]
            epoch_solutions = [[epoch_solution, solution_pchgs["cost"], solution_pchgs["prize"]]]
            found_at_ns = 0
        else:
            epoch_solutions = list(solve_static_vrptw(args, epoch_instance_dispatch, executable,
                                   time_limit=int(epoch_tlim - 2 - time_passed_instance_start_of_evaluation),
                                   seed=solver_seed))
            epoch_solution, cost, found_at_ns = epoch_solutions[-1]
            assert len(epoch_solutions) > 0, f"No solution found during epoch {observation['current_epoch']}"
            # Map HGS solution to indices of corresponding requests
            epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution if len(route) > 0]
        print("remaining_time after hgs: {}".format(str(epoch_tlim - (time.perf_counter() - epoch_evaluation_start_time))))

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)
        print("Default time limit env: {}".format(env.default_epoch_tlim))
        print(f'Cost of solution: {reward}', file=sys.stderr)
        assert info['error'] is None, info['error']
        if args.strategy not in ["NeuralNetwork", "Linear", "GraphNeuralNetwork", "GraphNeuralNetwork_sparse"]:
            assert cost is None or reward == -cost, "Reward should be negative cost of solution"
            assert not info['error'], f"Environment error: {info['error']}"
        cost = reward

        total_reward += reward
        solutions.extend((create_solution_from_dynamic(sol[0], sol[1], current_epoch, sol[2]) for sol in epoch_solutions))
        best_solution.append((epoch_solution, float(cost)))

        if done:
            break
        epoch_idx += 1

    best_solution = Solution(
        [Route(list(map(lambda x: int(x), route)), epoch=int(epoch + 1)) for epoch, (epoch_sol, _) in
         enumerate(best_solution) for route in epoch_sol], cost=int(sum(x[1] for x in best_solution)),
        found_after_ms=found_at_ns // 1000)

    return best_solution, solutions


def dump_solutions(writer: TextIO, solutions: List[Solution]):
    serialized = dataclass_factory.Factory().dump(solutions, List[Solution])
    json.dump(serialized, writer, cls=tools.NumpyJSONEncoder)


def run_evaluation_catch_error(args, instance):
    try:
        run_evaluation(copy.deepcopy(args), instance)
    except Exception as e:
        print(e)


def run_evaluation(args, instance):
    args.instance = instance
    args = util.set_additional_args(args)
    if os.path.isfile(args.result_directory + "/best-sol.json"):
        print(args.result_directory + "/best-sol.json already exists.")
        return

    if args.strategy in ["NeuralNetwork", "Linear", "GraphNeuralNetwork", "GraphNeuralNetwork_sparse"]:
        args.hgs_executable = str(next(Path("../pchgs/build").rglob("PCHGS*")))
    else:
        args.hgs_executable = "./" + str(next(Path(".").rglob("genvrp*")))

    util.create_directories([args.result_directory])

    strategy = STRATEGIES.get(args.strategy, None)

    # Set up environment (instance)
    instance = tools.read_vrplib(args.instance)
    env = VRPEnvironment(seed=args.instance_seed, instance=instance, epoch_tlim=args.time_limit + 1000, is_static=False)

    if args.strategy == "oracle":
        sols = run_oracle(args, env, executable=args.hgs_executable)
        best_sol = min(sols, key=lambda x: x.cost)
    else:
        best_sol, sols = run_strategy(args, env, args.hgs_executable, args.solver_seed, strategy, instance=instance)

    print(f'Found best solution with cost {best_sol.cost}', file=sys.stderr)

    with open(args.result_directory + "/solutions.json", 'w') as output:
        dump_solutions(output, sols)
    with open(args.result_directory + "/seeds.json", "w") as output:
        json.dump(dict(instance_seed=args.instance_seed, solver_seed=args.solver_seed), output)
    with open(args.result_directory + "/instance.txt", "w") as output:
        output.writelines([args.instance])
    with open(args.result_directory + "/best-sol.json", "w") as output:
        dump_solutions(output, [best_sol])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = config.parser.parse_args()
    print(args)
    if args.instance_directory is None:
        run_evaluation(copy.deepcopy(args), args.instance)
    else:
        with Pool(int(mp.cpu_count()) - 2) as pool:
            pool.map(partial(run_evaluation_catch_error, copy.deepcopy(args)), [args.instance_directory + file for file in os.listdir(args.instance_directory)])