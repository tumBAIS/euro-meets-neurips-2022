import tensorflow as tf
import numpy as np


class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GraphConvLayer, self).__init__()

        fnn_layers = []
        fnn_layers.append(tf.keras.layers.Dense(10, activation='relu'))
        fnn_layers.append(tf.keras.layers.Dense(10, use_bias=False))
        self.ffn_prepare = tf.keras.Sequential(fnn_layers)

        fnn_layers = []
        fnn_layers.append(tf.keras.layers.Dense(20, activation='relu'))
        fnn_layers.append(tf.keras.layers.Dense(10, use_bias=False))
        self.update_fn = tf.keras.Sequential(fnn_layers)

    def prepare(self, node_repesentations, edge_features):
        # node_repesentations shape is [num_edges, embedding_dim].
        input_features = tf.concat([edge_features, node_repesentations], axis=1)
        messages = self.ffn_prepare(input_features)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = tf.shape(node_repesentations)[0]
        aggregated_message = tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)
        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        # Concatenate the node_repesentations and aggregated_messages.
        h = tf.concat([node_repesentations, aggregated_messages], axis=1)

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_features = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        neighbour_indices, node_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)
        #edge_representation = tf.gather(edge_features, (neighbour_indices, node_indices))

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_features)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(node_indices, neighbour_messages, node_repesentations)
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)

def create_ffn_output():
    fnn_layers = []
    fnn_layers.append(tf.keras.layers.Dense(10, activation='relu'))
    fnn_layers.append(tf.keras.layers.Dense(10, activation='relu'))
    fnn_layers.append(tf.keras.layers.Dense(10, activation='relu'))
    fnn_layers.append(tf.keras.layers.Dense(1, use_bias=False))
    return tf.keras.Sequential(fnn_layers)

def create_ffn_preprocess():
    fnn_layers = []
    fnn_layers.append(tf.keras.layers.Dense(10, activation='relu'))
    fnn_layers.append(tf.keras.layers.Dense(10, use_bias=False))
    return tf.keras.Sequential(fnn_layers)


class GraphNeuralNetwork():
    def __init__(self, learning_rate=None):
        super().__init__()
        self.model = None
        if learning_rate is not None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    def create_model(self, input_dimension, input_dimension_edges):

        # Create a process layer.
        preprocess = create_ffn_preprocess()

        # Create the first GraphConv layer.
        conv1 = GraphConvLayer()
        # Create the second GraphConv layer.
        conv2 = GraphConvLayer()

        # Create a postprocess layer.
        postprocess = create_ffn_output()


        node_features = tf.keras.layers.Input((input_dimension), dtype="float32", name="node_features")
        edge_features = tf.keras.layers.Input((input_dimension_edges), dtype="float32", name="edge_features")
        edges = tf.keras.layers.Input((2), dtype="int32", name="edges")

        edges_trans = tf.transpose(edges)
        # Preprocess the node_features to produce node representations.
        x = preprocess(node_features)
        # Apply the first graph conv layer.
        x1 = conv1((x, edges_trans, edge_features))
        # Apply the second graph conv layer.
        x2 = conv2((x1, edges_trans, edge_features))
        # Postprocess node embedding.
        y = postprocess(x2)


        model = tf.keras.Model(
            inputs=[node_features, edge_features, edges],
            outputs=[y],
        )

        self.model = model

    def save_model(self, directory, count):
        self.model.save(directory + "_iteration-" + str(count))

    def predict_profits(self, features, edge_features, edges):
        profits = self.model([features.values, edge_features.values, np.array(edges)])
        profits = np.squeeze(profits).copy()
        if len(profits.shape) > 0:
            profits[0] = 0
        else:
            profits = np.array([0])
        if profits[0] != 0:
            raise Exception("Profit of depot is not 0")
        return profits

    def grad_optimize(self, n, n_hat_mean, features, edge_features, edges):
        with tf.GradientTape() as tape_node:
            profits = tf.squeeze(self.model([features.values, edge_features.values, np.array(edges)], training=True))
            correct_objective_profits = tf.math.reduce_sum(tf.multiply(profits, tf.squeeze(n.astype("float32"))))
            predicted_objective_profits = tf.math.reduce_sum(tf.multiply(profits, tf.squeeze(n_hat_mean.astype("float32"))))
            loss_node = tf.math.subtract(predicted_objective_profits, correct_objective_profits)
        self.optimizer.apply_gradients(zip(tape_node.gradient(loss_node, self.model.trainable_variables), self.model.trainable_variables))


class GraphNeuralNetwork_sparse(GraphNeuralNetwork):
    def __init__(self, learning_rate=None):
        super().__init__(learning_rate)


class Linear():
    def __init__(self, learning_rate=None):
        self.model = None
        if learning_rate is not None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def create_model(self, input_dimension):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dimension,)))
        model.add(tf.keras.layers.Dense(1, use_bias=False))
        self.model = model

    def save_model(self, directory, count):
        self.model.save(directory + "_iteration-" + str(count))

    def predict_profits(self, features, edge_features, edges):
        profits = self.model.predict(features.values, verbose=0)
        profits = np.squeeze(profits)
        if len(profits.shape) > 0:
            profits[0] = 0
        else:
            profits = np.array([0])
        if profits[0] != 0:
            raise Exception("Profit of depot is not 0")
        return profits

    def grad_optimize(self, n, n_hat_mean, features, edge_features=None, edges=None):
        with tf.GradientTape() as tape_node:
            profits = tf.squeeze(self.model(features.values, training=True))
            correct_objective_profits = tf.math.reduce_sum(tf.multiply(profits, tf.squeeze(n.astype("float32"))))
            predicted_objective_profits = tf.math.reduce_sum(tf.multiply(profits, tf.squeeze(n_hat_mean.astype("float32"))))
            loss = tf.math.subtract(predicted_objective_profits, correct_objective_profits)
        self.optimizer.apply_gradients(zip(tape_node.gradient(loss, self.model.trainable_variables), self.model.trainable_variables))

class NeuralNetwork():
    def __init__(self, learning_rate=None):
        self.model = None
        if learning_rate is not None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def create_model(self, input_dimension):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dimension,)))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(1, use_bias=False))
        self.model = model

    def save_model(self, directory, count):
        self.model.save(directory + "_iteration-" + str(count))

    def predict_profits(self, features, edge_features, edges):
        profits = self.model.predict(features.values, verbose=0)
        profits = np.squeeze(profits)
        if len(profits.shape) > 0:
            profits[0] = 0
        else:
            profits = np.array([0])
        if profits[0] != 0:
            raise Exception("Profit of depot is not 0")
        return profits

    def grad_optimize(self, n, n_hat_mean, features, edge_features, edges):
        with tf.GradientTape() as tape_node:
            profits = tf.squeeze(self.model(features.values, training=True))
            correct_objective_profits = tf.math.reduce_sum(tf.multiply(profits, tf.squeeze(n.astype("float32"))))
            predicted_objective_profits = tf.math.reduce_sum(tf.multiply(profits, tf.squeeze(n_hat_mean.astype("float32"))))
            loss = tf.math.subtract(predicted_objective_profits, correct_objective_profits)
        self.optimizer.apply_gradients(zip(tape_node.gradient(loss, self.model.trainable_variables), self.model.trainable_variables))
