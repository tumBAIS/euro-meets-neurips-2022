import json
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(loss_curve, accuracy_curve_n, default_accuracy_n, title=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(list(range(len(loss_curve))), loss_curve)
    ax1.set_title("Loss")
    ax1.set_ylim(bottom=0, top=max(np.array(loss_curve)) + 100)
    ax2.plot(list(range(len(accuracy_curve_n))), accuracy_curve_n)
    ax2.hlines(default_accuracy_n, xmin=0, xmax=len(loss_curve), colors="red")
    ax2.set_title("Accuracy Nodes")
    if title is not None:
        fig.suptitle(title + "\n" + "\n" + "(The red line specifies the default accuracy)", fontsize=7)
    fig.supxlabel('training epochs')
    plt.tight_layout()
    plt.show()


def get_learning_curve(learning_name):
    parameter_evolution = json.load(open('{}'.format(learning_name), 'r'))
    plot_learning_curves(loss_curve=parameter_evolution["overall_loss"],
                         accuracy_curve_n=parameter_evolution["overall_accuracy_n"],
                         default_accuracy_n=parameter_evolution["default_accuracy_n"],
                         title=learning_name)



if __name__ == "__main__":
    get_learning_curve("./learning_evaluation_NeuralNetwork_samples-50_instances-15_runtime-3600_featureset-dynamicstatic.json")
