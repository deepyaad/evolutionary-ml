from pprint import pprint as prtty
import numpy as np 
import random as rnd
from evo_v5 import Environment
from objectives import *
from mutators import *
from solution import Solution
import uuid


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

        # create random configurations of small pure recurrent neural networks 
        # no dense, pooling, convolution, or normalization layers for initial population
        hidden_layer_count = rnd.randint(1, 3)
        hidden_layers, layer_names, specifications, outputs = [], [], [], []
        input_size = feature_shape
        activations = [
            'celu', 'elu', 'gelu', 'hard_sigmoid', 'hard_shrink', 'hard_tanh', 'hard_silu', 
            'leaky_relu', 'linear', 'mish', 'relu', 'selu', 'silu', 
            'sigmoid', 'softmax', 'softplus', 'softsign', 'soft_shrink', 'swish', 'tanh',
            'tanh_shrink'
        ]
        for _ in range(hidden_layer_count):

            # randomize activation function and units
            activation = rnd.choice(activations)
            recurrent_activation = rnd.choice(activations)
            units = rnd.randint(8, 48)
            random_specs = {
                'activation': activation,
                'recurrent_activation': recurrent_activation,
                'units': units
            }
            rnn_archs = ['LSTM', 'SimpleRNN', 'GRU']    

            # create hidden layers
            if _ == hidden_layer_count - 1:
                # create last hidden layer
                            
                last_hidden_layer, name, specs, output_size = create_layer(input_size, rnn_archs, last_layer=True, specs=random_specs)

                # add last hidden layer to list of hidden layers
                hidden_layers.append(last_hidden_layer)
                layer_names.append(name)
                specifications.append(specs)
                outputs.append(output_size)

            else:
                # intialize models with only recurrent layers
                layer, name, specs, output_size = create_layer(input_size, rnn_archs, last_layer=False, specs=random_specs)
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
                'kl_divergence'
            ]),
            'optimizer': rnd.choice([
                'adamax', 'adadelta','ftrl', 'lamb',  'nadam', 'rmsprop', 
                'sgd', 'adagrad',  'lion',  'adamw', 'adafactor', 'adam'                                
            ]),

            'epochs': rnd.randint(3, 60),
            'batch_size': rnd.randint(32, 512),

            # input data specifications
            'input_size': input_size,
            'output_size': class_count,
            'feature_shape': feature_shape,
            'class_count': class_count,
            'labels_inorder': labels_inorder,

            # genetic information
            'id': uuid.uuid4(),
            'parent_id': 'start',
            'mutator': 'randomization'
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
    env.evolve(n=10**9, dom=30, status=30, viol=10, time_limit=300, reset=True, historical_pareto=False) # 28800 seconds = 8 hours


if __name__ == "__main__":
    main()