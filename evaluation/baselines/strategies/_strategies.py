import numpy as np
from evaluation import tools
from evaluation.environment import State
from evaluation.src import util
from features.FeatureComputer import format_features


def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items():
        if key == 'capacity':
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res


def _greedy(observation: State, rng: np.random.Generator):
    return {
        **observation,
        'must_dispatch': np.ones_like(observation['must_dispatch']).astype(np.bool8)
    }


def _lazy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    return _filter_instance(observation, mask)


def _random(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.5, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)


def _SL_NN(args, observation: State, model, instance, static_info):
    def update_profits_to_ensure_must_dispatch(profits, costs, must_dispatch):
        max_from_dest = np.max(costs, axis=1)
        max_to_dest = np.max(costs, axis=0)
        max = 2 * (max_from_dest + max_to_dest)
        profits[must_dispatch] = max[must_dispatch]
        return profits

    feature_dict = format_features(args=args, observation=observation, instance=instance, static_info=static_info)

    profits = model.predict_profits(features=feature_dict["features"], edge_features=feature_dict["edge_features"], edges=feature_dict["graph_edges"])
    costs = observation["epoch_instance"]["duration_matrix"]
    profits = update_profits_to_ensure_must_dispatch(profits, costs, observation["epoch_instance"]["must_dispatch"])
    profits = util.make_to_ints(profits)
    pchgs_instance = tools.format_pchgs_instance_as_json(args, instance=observation["epoch_instance"], profits=profits)
    return {"request_idx": observation["epoch_instance"]["request_idx"],
            "pchgs_instance": pchgs_instance,
            "epoch_data": observation, "profits": profits}



STRATEGIES = dict(
    greedy=_greedy,
    lazy=_lazy,
    random=_random,
    NeuralNetwork=_SL_NN,
    Linear=_SL_NN,
    GraphNeuralNetwork=_SL_NN,
    GraphNeuralNetwork_sparse=_SL_NN,
    rolling_horizon=_greedy,
    monte_carlo=_greedy)
