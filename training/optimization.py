from prediction import NeuralNetwork, GraphNeuralNetwork, Linear, GraphNeuralNetwork_sparse
import json
from src import util
from evaluation import tools
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
import numpy as np


def loss_for_perturbation(kwargs, perturbation_num):
    args, training_instance, costs, profits = kwargs
    profits_perturbed = util.apply_perturbation(args, profits)
    profits_perturbed = util.make_to_ints(profits_perturbed)
    pchgs_instance = tools.format_pchgs_instance_as_json(args, instance=training_instance["epoch_instance_numpy"], profits=profits_perturbed)
    solution_pchgs = tools.solve_pchgs(args=args, instance=pchgs_instance, executable=args.pchgs_executable, time_limit=args.time_limit, seed=1)
    cost_hat, y_hat, n_hat = util.decode_solution(solution_pchgs, edges=training_instance["edges"], nodes=training_instance["nodes"])
    loss_hat = np.sum(profits_perturbed[n_hat]) - np.sum(costs.flatten(order="C")[y_hat])
    return loss_hat, y_hat, n_hat


class Optimizer():
    def __init__(self, args, num_features, num_edge_features):
        self.args = args
        if self.args.predictor == "NeuralNetwork":
            model = NeuralNetwork(self.args.learning_rate)
            model.create_model(input_dimension=num_features)
        elif self.args.predictor == "Linear":
            model = Linear(self.args.learning_rate)
            model.create_model(input_dimension=num_features)
        elif self.args.predictor == "GraphNeuralNetwork":
            model = GraphNeuralNetwork(self.args.learning_rate)
            model.create_model(input_dimension=num_features, input_dimension_edges=num_edge_features)
        elif self.args.predictor == "GraphNeuralNetwork_sparse":
            model = GraphNeuralNetwork_sparse(self.args.learning_rate)
            model.create_model(input_dimension=num_features, input_dimension_edges=num_edge_features)
        else:
            raise Exception("Wrong predictor defined")
        self.model = model
        self.overall_accuracy_n = []
        self.overall_loss = []
        self.default_accuracy_n = []

    def save_learning_evaluation(self):
        json.dump({"default_accuracy_n": self.default_accuracy_n,
                   "overall_loss": self.overall_loss,
                   "overall_accuracy_n": self.overall_accuracy_n},
        open('./learning_evaluation_{}.json'.format(self.args.model_name), 'w'))

    def loss(self, training_instance):
        _, y, n = util.decode_solution(solution=training_instance["epoch_solution"], edges=training_instance["edges"], nodes=training_instance["nodes"])
        profits = self.model.predict_profits(features=training_instance["features"], edge_features=training_instance["edge_features"], edges=training_instance["graph_edges"])
        costs = training_instance["duration_matrix"]
        with Pool(mp.cpu_count()-2) as pool:
            saver = pool.map(partial(loss_for_perturbation, (self.args, training_instance, costs, profits)), list(range(self.args.num_perturbations)))
        profits = util.make_to_ints(profits)
        loss = np.sum(profits[n]) - np.sum(costs.flatten(order="C")[y])
        loss_saver = [value[0] for value in saver]
        y_saver = [value[1] for value in saver]
        n_saver = [value[2] for value in saver]
        loss_hat_mean = np.mean(loss_saver)
        n_hat_mean = np.mean(n_saver, axis=0)
        self.model.grad_optimize(np.array(n), n_hat_mean, features=training_instance["features"], edge_features=training_instance["edge_features"], edges=training_instance["graph_edges"])
        return loss_hat_mean - loss, n, n_saver

    def train(self, training_instances):
        print("Start training ...")
        self.default_accuracy_n = util.calculate_default_accuracy(training_instances=training_instances)
        for epoch in range(self.args.num_training_epochs):
            loss_value_instance = []
            accuracy_n_instance = []
            for training_instance in training_instances:
                loss_value, n, n_saver = self.loss(training_instance)
                accuracy_n = util.calculate_accuracy_mean(n=np.array(n), n_hat=np.array(n_saver))
                loss_value_instance.append(loss_value)
                accuracy_n_instance.append(accuracy_n)
            self.overall_loss.append(np.mean(loss_value_instance))
            self.overall_accuracy_n.append(np.mean(accuracy_n_instance))
            self.model.save_model(directory=self.args.dir_models + self.args.model_name, count=epoch)
            self.save_learning_evaluation()
            print("Epoch: {} ----> Loss: {} ---- Accuracy_n: {}".format(epoch, np.mean(loss_value_instance), np.mean(accuracy_n_instance)))
