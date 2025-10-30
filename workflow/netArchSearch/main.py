from pprint import pprint as prtty
import numpy as np 
import random as rnd
from evo_v5 import Environment
from objectives import *
from profiler import Profiler
from agents import *
from solution import Solution
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_preprocessing():

    # load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split into train, test and validation sets
    x_train, x_eval,  y_train, y_eval = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_eval, y_eval, test_size=0.5, random_state=42)

    # store preprocessed data
    data = {
        'train_features': x_train, 'train_labels': y_train,
        'val_features': x_val, 'val_labels': y_val,
        'test_features': x_test, 'test_labels': y_test
    }

    return data

print('data preprocessing function defined\n\n')

def main():

    # intialize data for model development
    data = data_preprocessing()
    class_count = len(np.unique(data['train_labels']))
    feature_count = data['train_features'].shape[1]
    print(f'class count: {class_count}, feature count: {feature_count}\n\n')

    # initialize environment
    env = Environment()

    # register objectives
    env.add_fitness_criteria("training_time", training_time)
    env.add_fitness_criteria("total_layers", total_layers)
    env.add_fitness_criteria("total_nodes", total_nodes)
    env.add_fitness_criteria("loss", loss)
    env.add_fitness_criteria("accuracy", accuracy)
    env.add_fitness_criteria("binary_accuracy", binary_accuracy)
    env.add_fitness_criteria("cosine_similarity", cosine_similarity)
    env.add_fitness_criteria("true_positive_rate", true_positive_rate)
    env.add_fitness_criteria("true_negative_rate", true_negative_rate)
    env.add_fitness_criteria("false_positive_rate", false_positive_rate)
    env.add_fitness_criteria("false_negative_rate", false_negative_rate)
    env.add_fitness_criteria("mean_squared_error", mean_squared_error)
    env.add_fitness_criteria("recall", recall)
    env.add_fitness_criteria("precision", precision)
    env.add_fitness_criteria("specificity", specificity)
    env.add_fitness_criteria("auc", auc)
    env.add_fitness_criteria("sensitivity", sensitivity)

    # register agents
    env.add_agent("add_layer", add_layer, 1)
    env.add_agent("remove_layer", remove_layer, 1)
    env.add_agent("grow_layer", grow_layer, 1)
    env.add_agent("shrink_layer", shrink_layer, 1)
    env.add_agent("change_activation", change_activation, 1)
    env.add_agent("change_optimizer", change_optimizer, 1)
    env.add_agent("change_units_per_layer", change_units_per_layer, 1)
    env.add_agent("change_epochs", change_epochs, 1)
    env.add_agent("change_batch_size", change_batch_size, 1)
    env.add_agent("change_loss_func", change_loss_func, 1)

    # register dataset
    env.add_dataset(data, class_count)

    print('environment initialized\n\n')

    # intialize solutions
    for _ in range(10):

        # create random hyperparameter dictionary
        hidden_layer_count = rnd.randint(2, 16)
        hyperparams = {
            'loss_function': rnd.choice(['binary_crossentropy', 'binary_focal_crossentropy', 'cosine_similarity']),
            'hidden_layer_count': hidden_layer_count, 
            'units_per_hidden_layer': [rnd.randint(16, 128) for _ in range(hidden_layer_count)],
            'activation_per_hidden_layer': [rnd.choice(['relu', 'tanh', 'sigmoid']) for _ in range(hidden_layer_count)],
            'optimizer': rnd.choice(['adam', 'sgd', 'rmsprop']),
            'epochs': rnd.randint(5, 20),
            'batch_size': rnd.choice([16, 32, 64, 128]),
            'feature_count': feature_count,
            'class_count': class_count,
        }
        sol = Solution(hyperparams)
        sol.develop_model(data)
        env.add_solution(sol)


    # run the optimizer
    env.evolve(10**9, dom=100, status=2000, time_limit=300, reset=True)
    env.summarize(source='ananda', with_details=False)


    # summarize results
    Profiler.report()
    prtty(env.pop)


if __name__ == "__main__":
    main()