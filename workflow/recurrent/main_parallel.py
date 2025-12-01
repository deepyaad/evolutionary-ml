"""
Parallel evolution main script using RunPod serverless.
This script initializes solutions and runs parallel evolution.
"""
from pprint import pprint as prtty
import numpy as np 
import random as rnd
from evo_v5 import Environment
from objectives import *
from mutators import *
from solution import Solution
from parallel_evolution import RunPodEvolutionClient, convert_result_to_solution
import uuid
import os


def main():
    # Get RunPod configuration from environment or file
    runpod_endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    runpod_api_key = os.getenv('RUNPOD_API_KEY')
    
    # If API key not in environment, try to read from api-key.txt file
    if not runpod_api_key:
        api_key_path = '../../api-key.txt'
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                runpod_api_key = f.read().strip()
            print("Loaded RunPod API key from api-key.txt")
        else:
            # Try current directory
            api_key_path = 'api-key.txt'
            if os.path.exists(api_key_path):
                with open(api_key_path, 'r') as f:
                    runpod_api_key = f.read().strip()
                print("Loaded RunPod API key from api-key.txt")
    
    if not runpod_endpoint_id:
        raise ValueError("RUNPOD_ENDPOINT_ID environment variable must be set")
    
    if not runpod_api_key:
        raise ValueError("RUNPOD_API_KEY must be set in environment variable or api-key.txt file")
    
    # Initialize data for model development
    data = np.load("../../datasets/spotify_dataset.npz", allow_pickle=True)
    labels_inorder = data['labels_inorder']
    _, class_count = data['train_labels'].shape
    feature_shape = data['train_features'].shape[1:]

    # Output dataset information
    print(f"train set: {data['train_features'].shape}")
    print(f"validation set: {data['val_features'].shape}")
    print(f"test set: {data['test_features'].shape}")
    print(f'class count: {class_count}, feature shape: {feature_shape}', '\n\n')
    print(f'classes (in encoding order): {labels_inorder}')

    # Initialize environment
    env = Environment()

    # Initialize objectives to minimize
    env.add_fitness_criteria("unfairness", unfairness)
    env.add_fitness_criteria("misclassification", misclassification)
    env.add_fitness_criteria("complexity", complexity)
    env.add_fitness_criteria("resource_utilization", resource_utilization)

    # Register agents
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
    
    # Register dataset
    env.add_dataset(data, class_count, feature_shape)
    print('environment initialized\n\n')

    # Initialize solutions (10 random configs)
    configurations = []
    for _ in range(10):
        # Create random configurations of small pure recurrent neural networks 
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
            # Randomize activation function and units
            activation = rnd.choice(activations)
            recurrent_activation = rnd.choice(activations)
            units = rnd.randint(8, 48)
            random_specs = {
                'activation': activation,
                'recurrent_activation': recurrent_activation,
                'units': units
            }
            rnn_archs = ['LSTM', 'SimpleRNN', 'GRU']    

            # Create hidden layers
            if _ == hidden_layer_count - 1:
                # Create last hidden layer
                last_hidden_layer, name, specs, output_size = create_layer(input_size, rnn_archs, last_layer=True, specs=random_specs)
                hidden_layers.append(last_hidden_layer)
                layer_names.append(name)
                specifications.append(specs)
                outputs.append(output_size)
            else:
                # Initialize models with only recurrent layers
                layer, name, specs, output_size = create_layer(input_size, rnn_archs, last_layer=False, specs=random_specs)
                hidden_layers.append(layer)
                layer_names.append(name)
                specifications.append(specs)
                outputs.append(output_size)
                input_size = output_size

        configuration = {
            # Architecture specifications
            'hidden_layer_count': hidden_layer_count, 
            'hidden_layers': hidden_layers,
            'layer_names': layer_names,
            'layer_specifications': specifications,
            'neurons_per_layer': outputs,

            # Hyperparameters
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

            # Input data specifications
            'input_size': input_size,
            'output_size': class_count,
            'feature_shape': feature_shape,
            'class_count': class_count,
            'labels_inorder': labels_inorder,

            # Genetic information
            'id': uuid.uuid4(),
            'parent_id': 'start',
            'mutator': 'randomization'
        }
        
        prtty(configuration)
        sol = Solution(configuration)
        configurations.append(sol)

    print('architectures and hyperparameters defined\n\n') 

    # Train initial solutions in parallel
    print("Training initial 10 solutions in parallel on RunPod...")
    client = RunPodEvolutionClient(runpod_endpoint_id, runpod_api_key)
    data_path = '../../datasets/spotify_dataset.npz'
    
    results = client.train_solutions_parallel(configurations, data_path, timeout=3600)
    
    # Add successful solutions to environment
    successful = 0
    for result in results:
        sol = convert_result_to_solution(result)
        if sol is not None:
            env.add_solution(sol)
            successful += 1
    
    print(f'Successfully trained and added {successful}/10 initial solutions\n\n')

    # Run parallel evolution
    # Configuration for high throughput:
    # - batch_size=50: Generate 50 children per generation
    # - With 50 parallel workers, each generation takes ~4.5 minutes
    # - In 30 minutes: ~6 generations = 300 new solutions + 10 initial = 310 total
    env.evolve_parallel(
        batch_size=50,  # Adjust based on your RunPod capacity
        generations=None,  # Run until time_limit
        dom=30,
        viol=10,
        status=30,
        time_limit=1800,  # 30 minutes for testing (use 28800 for 8 hours)
        reset=True,
        historical_pareto=False,
        runpod_endpoint_id=runpod_endpoint_id,
        runpod_api_key=runpod_api_key,
        data_path=data_path
    )


if __name__ == "__main__":
    main()

