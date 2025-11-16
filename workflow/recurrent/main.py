from pprint import pprint as prtty
import numpy as np 
import random as rnd
import os
from evo_v5 import Environment
from objectives import *
from mutators import *
from solution import Solution
from data_loader import LazyDataLoader



def main():

    # Initialize data for model development
    # Try to use LazyDataLoader first (for separate feature files)
    # Fall back to legacy consolidated format if needed
    data_path = "../../datasets"
    
    # Check if we have separate feature files (new format from preprocessing notebook)
    # or a consolidated file (legacy format)
    consolidated_path = "../../datasets/feature_representations.npz"
    
    if os.path.exists(consolidated_path):
        # Legacy format - load all features at once (memory intensive)
        print("Loading data in legacy consolidated format...")
        data = np.load(consolidated_path, allow_pickle=True)
        labels_inorder = data['labels_inorder']
        _, class_count = data['train_labels'].shape
        feature_shape = data['stft_train_features'].shape[1:]
        
        # output dataset information
        print(f"train set: {data['stft_train_features'].shape}")
        print(f"validation set: {data['stft_val_features'].shape}")
        print(f"test set: {data['stft_test_features'].shape}")
        print(f'class count: {class_count}, feature shape: {feature_shape}', '\n\n')
        print(f'classes (in encoding order): {labels_inorder}')
        print("WARNING: Using legacy format - all features loaded into memory!")
        print("Consider using separate feature files for memory efficiency.\n")
    else:
        # New format - use LazyDataLoader for memory efficiency
        print("Loading data using LazyDataLoader (memory-efficient)...")
        data = LazyDataLoader(data_path, mode='separate')
        labels_inorder = data.labels_inorder
        class_count = data.class_count
        
        # Get feature shape from first available feature
        available_features = [f for f in LazyDataLoader.AVAILABLE_FEATURES 
                            if data.is_feature_available(f)]
        if not available_features:
            raise ValueError("No available features found!")
        
        # Use first available feature for initial shape
        first_feature = available_features[0]
        feature_shape = data.get_feature_shape(first_feature)
        
        # output dataset information
        print(f"Available features: {available_features}")
        print(f'class count: {class_count}, default feature shape: {feature_shape}', '\n\n')
        print(f'classes (in encoding order): {labels_inorder}')
        print("Using lazy loading - features will be loaded on-demand.\n")

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
    env.add_agent("change_feature_selection", change_feature_selection)  # NEW: Feature evolution!
    
    # register dataset
    if isinstance(data, LazyDataLoader):
        env.add_dataset(data)  # LazyDataLoader provides metadata automatically
    else:
        env.add_dataset(data, class_count, feature_shape)  # Legacy format
    print('environment initialized\n\n')

    # intialize solutions
    configurations = []
    for _ in range(10):

        # get random feature representation - use available features
        if isinstance(data, LazyDataLoader):
            available_features = [f for f in LazyDataLoader.AVAILABLE_FEATURES 
                                if data.is_feature_available(f)]
            feature_selection = rnd.choice(available_features)
            # Get feature shape for selected feature
            current_feature_shape = data.get_feature_shape(feature_selection)
        else:
            # Legacy format
            feature_selection = rnd.choice([
                'stft', 'mel_specs', 'mfccs', 'mctct', 'parakeet', 
                'seamlessM4T', 'whisper'
            ])
            current_feature_shape = feature_shape

        # create random configurations of recurrent neural networks
        hidden_layer_count = rnd.randint(2, 4)
        hidden_layers, layer_names, specifications, outputs = [], [], [], []
        input_size = current_feature_shape
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
                'kl_divergence',
            ]),
            'optimizer': rnd.choice([
                'adamax', 'adadelta','ftrl', 'lamb',  'nadam', 'rmsprop', 
                'sgd', 'adagrad',  'lion',  'adamw', 'adafactor', 'adam'                                
            ]),

            'epochs': rnd.randint(3,200),   # TODO: allow model to add more epochs
            'batch_size': rnd.randint(32, 512),

            # feature selection
            'feature_selection': feature_selection,

            # input data specifications
            'input_size': input_size,
            'output_size': class_count,
            'feature_shape': current_feature_shape,
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