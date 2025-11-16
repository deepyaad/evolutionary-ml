from pprint import pprint as prtty
import numpy as np 
import random as rnd
from evo_v5 import Environment
from objectives import *
from mutators import *
from solution import Solution



def main():

    # intialize data for model development
    data = np.load("../../datasets/spotify_dataset.npz", allow_pickle=True)
    labels_inorder = data['labels_inorder']
    _, class_count = data['train_labels'].shape
    feature_shape = data['train_features'].shape[1:]

    # output dataset information
    print(f"train set: {data['train_features'].shape}")
    print(f"validation set: {data['val_features'].shape}")
    print(f"test set: {data['test_features'].shape}")
    print(f'class count: {class_count}, feature shape: {feature_shape}', '\n\n')
    print(f'classes (in encoding order): {labels_inorder}')

    # initialize environment
    env = Environment()

    # initialize objectives to minimize
    env.add_fitness_criteria("unfairness", unfairness)
    env.add_fitness_criteria("misclassification", misclassification)
    env.add_fitness_criteria("complexity", complexity)
    env.add_fitness_criteria("resource_utilization", resource_utilization)

    # register agents
    env.add_agent("add_layer", add_layer)
    env.add_agent("remove_layer", remove_layer)
    env.add_agent("swap_layers", swap_layers)
    env.add_agent("grow_layer", grow_layer)
    env.add_agent("shrink_layer", shrink_layer)
    env.add_agent("change_activation", change_activation)
    env.add_agent("change_optimizer", change_optimizer)
    env.add_agent("change_epochs", change_epochs)
    env.add_agent("change_batch_size", change_batch_size)
    env.add_agent("change_loss_func", change_loss_func)
    
    # register dataset
    env.add_dataset(data, class_count, feature_shape)
    print('environment initialized\n\n')

    # intialize solutions
    configurations = []
    for _ in range(10):

        # create random configurations of recurrent neural networks
        hidden_layer_count = rnd.randint(2, 4)
        hidden_layers, layer_names, specifications, outputs = [], [], [], []
        input_size = feature_shape
        for _ in range(hidden_layer_count):

            # create hidden layers
            if _ == hidden_layer_count - 1:
                # create last hidden layer
                valid_last_layers = ['LSTM', 'SimpleRNN', 'GRU','GlobalAveragePooling1D', 'GlobalMaxPooling1D']
                last_hidden_layer, name, specs, output_size = create_layer(input_size, valid_last_layers, last_layer=True)

                # add last hidden layer to list of hidden layers
                hidden_layers.append(last_hidden_layer)
                layer_names.append(name)
                specifications.append(specs)
                outputs.append(output_size)

            else:
                # intialize models with only recurrent layers
                layer, name, specs, output_size = create_layer(input_size, ['LSTM', 'SimpleRNN', 'GRU'], last_layer=False)
                hidden_layers.append(layer)
                layer_names.append(name)
                specifications.append(specs)
                outputs.append(output_size)
                input_size = output_size

        configuration = {

            # architecture specifications
            'hidden_layer_count': hidden_layer_count, 
            'hidden_layers': hidden_layers,
            'layer_names': layer_names,
            'layer_specifications': specifications,
            'neurons_per_layer': outputs,

            # hyperparameters
            'loss_function': rnd.choice([
                'categorical_crossentropy', 'categorical_focal_crossentropy', 
                'kl_divergence', 'sparse_categorical_crossentropy'
            ]),
            'optimizer': rnd.choice([
                'adamax', 'adadelta','ftrl', 'lamb',  'nadam', 'rmsprop', 
                'sgd', 'adagrad',  'lion',  'adamw', 'adafactor', 'adam'                                
            ]),

            'epochs': rnd.randint(3,200),   # TODO: allow model to add more epochs
            'batch_size': rnd.randint(32, 512),

            # input data specifications
            'input_size': input_size,
            'output_size': class_count,
            'feature_shape': feature_shape,
            'class_count': class_count,
            'labels_inorder': labels_inorder
        }
        
        prtty(configuration)
        sol = Solution(configuration)
        configurations.append(sol)

    print('architectures and hyperparameters defined\n\n') 

    i = 1
    for sol in configurations:
        print('\n\n', f'training solution {i}')
        sol.develop_model(data)
        env.add_solution(sol)
        i += 1

    print('solutions trained and added to population\n\n')


    # run the optimizer
    env.evolve(n=10**9, dom=20, status=30, viol=10, time_limit=28800, reset=True, sync=30, historical_pareto=False)


if __name__ == "__main__":
    main()