from pprint import pprint as prtty
import numpy as np 
import random as rnd
from evo_v5 import Environment
from objectives import *
from agents import *
from solution import Solution



def main():

    # intialize data for model development
    data = np.load("../../datasets/spotify_dataset.npz", allow_pickle=True)
    labels_inorder = data['labels_inorder']
    _, class_count = data['train_labels'].shape
    feature_shape = data['train_features'].shape[1:]

    print(f"train set: {data['train_features'].shape}")
    print(f"validation set: {data['val_features'].shape}")
    print(f"test set: {data['test_features'].shape}")
    print(f'class count: {class_count}, feature shape: {feature_shape}', '\n\n')
    print(f'classes (in encoding order): {labels_inorder}')

    # initialize environment
    env = Environment()

    # register objectives
    env.add_fitness_criteria("development_time", development_time)
    # env.add_fitness_criteria("total_layers", total_layers)
    # env.add_fitness_criteria("total_nodes", total_nodes)
    env.add_fitness_criteria("loss", loss)
    env.add_fitness_criteria("accuracy", accuracy)
    env.add_fitness_criteria("f1", f1_score)
    env.add_fitness_criteria("precision", precision)

    # register agents
    env.add_agent("add_layer", add_layer)
    env.add_agent("remove_layer", remove_layer)
    env.add_agent("swap_layers", swap_layers)
    
    # env.add_agent("grow_layer", grow_layer)
    # env.add_agent("shrink_layer", shrink_layer)
    # env.add_agent("change_activation", change_activation)
    # env.add_agent("change_optimizer", change_optimizer)
    # env.add_agent("change_epochs", change_epochs)
    # env.add_agent("change_batch_size", change_batch_size)
    # env.add_agent("change_loss_func", change_loss_func)
    

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
            if _ == hidden_layer_count - 1:
                valid_last_layers = [
                    'LSTM', 'SimpleRNN', 'GRU',
                    'GlobalAveragePooling1D', 'GlobalMaxPooling1D'
                ]
                prev_layer_size = outputs[-1]
                last_hidden_layer, name, specs, output_size = create_layer(prev_layer_size, valid_last_layers, last_layer=True)
                hidden_layers.append(last_hidden_layer)
                layer_names.append(name)
                specifications.append(spec)
                outputs.append(output_size)
 
            else:
                layer, name, spec, output_size = create_layer(input_size, ['LSTM', 'SimpleRNN', 'GRU'], last_layer=False)
                hidden_layers.append(layer)
                layer_names.append(name)
                specifications.append(spec)
                outputs.append(output_size)
                input_size = output_size

        configuration = {
            
            # architecture
            'hidden_layer_count': hidden_layer_count, 
            'hidden_layers': hidden_layers,
            'layer_names': layer_names,
            'layer_specifications': specifications,
            'neurons_per_layer': outputs,

            # hyperparameters
            'loss_function': rnd.choice(['categorical_crossentropy', 'categorical_focal_crossentropy', 'kl_divergence']),
            'optimizer': rnd.choice([
                'adam', 'adamax', 'adadelta','ftrl', 'lamb',  'nadam', 'rmsprop', 'sgd', #, 'adagrad', 'adafactor', 'lion',  'adamw', 
            ]),
            'epochs': rnd.randint(3,18),
            'batch_size': rnd.randint(128, 450),

            # input data
            'feature_shape': feature_shape,
            'class_count': class_count,
        }
        
        print(configuration)
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
    env.evolve(n=10**9, dom=100, status=2000, time_limit=900, reset=True, sync=100, historical_pareto=False)


if __name__ == "__main__":
    main()