"""
@author: Ananda Francis
@file: manual.py: Manually design neural networks to build intuition on the neural architecture search problem.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import time
import pandas as pd
print('libraries imported')


from tensorflow import keras
from keras import layers
from keras.metrics import (
    Accuracy, BinaryAccuracy, CosineSimilarity,
    TruePositives, TrueNegatives, F1Score,
    MeanSquaredError, MeanAbsoluteError,
    Recall, Precision, AUC,
    FalsePositives, FalseNegatives,
    SensitivityAtSpecificity, SpecificityAtSensitivity
)
print('tensorflow imported')

'''
6 major types of deep neural networks are widely used in practice:
    * artificial neural networks for classification and regression tasks of tabular data
    * recurrent neural networks for language or time series tasks of sequential data
    * convolutional neural networks for computer vision tasks of spatial/image data

    * generative adversarial networks for generation tasks like image synthesis and data augmentation
    * deep belief networks for unsupervised learning tasks like feature extraction and dimensionality reduction
    * stacked auto-encoders for unsupervised learning and data compression tasks
'''

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

print('data preprocessing function defined')

def build_neural_network(type, model_info, class_count, feature_count):

    if type == 'ann':

        # extract hyperparameters
        layers_count = model_info['hyperparameters']['layers']
        units_per_layer = model_info['hyperparameters']['units']
        activation_function = model_info['hyperparameters']['activation']
        optimizer = model_info['hyperparameters']['optimizer']
        loss_function = model_info['hyperparameters']['loss']

        # store total nodes
        model_info['objectives']['total_nodes'] = units_per_layer * layers_count

        # build artificial neural network
        model = keras.Sequential()

        # initialize input layer
        model.add(layers.InputLayer(shape=(feature_count,)))

        # add hidden layers
        for _ in range(layers_count):
            model.add(layers.Dense(units_per_layer, activation=activation_function))

        # add output layer
        if class_count == 2: # binary classification
            model.add(layers.Dense(1, activation='sigmoid'))
        else: # multi-class classification
            model.add(layers.Dense(class_count, activation='softmax'))

        # compile the model
        model.compile(optimizer=optimizer,
                      loss=loss_function,
            metrics=[
                Accuracy(), BinaryAccuracy(), CosineSimilarity(),
                # F1Score(threshold=0.5),
                FalsePositives(), FalseNegatives(), TruePositives(), TrueNegatives(),
                MeanSquaredError(), Recall(), Precision(), AUC(),
                SensitivityAtSpecificity(specificity=0.9), SpecificityAtSensitivity(sensitivity=0.9)
            ]
        )

        # store model object
        model_info['model_object'] = model

        return model_info

    else:
        # value error for unsupported network types
        raise ValueError("Unsupported neural network type.")

print('neural network building function defined')

def evaluate_model_performance(data, model_info):
    """
    purpose: train and evaluate model object based on hyperparameters
    params: data - preprocessed dataset
            model_info - dictionary containing model hyperparameters and object
    output: updated model_info with performance metrics
    """

    model = model_info['model_object']
    batch_size = model_info['hyperparameters']['batch_size']
    epochs = model_info['hyperparameters']['epochs']

    # intialize timer
    start = time.time()

    # train the model
    model.fit(data['train_features'], data['train_labels'], epochs=epochs, batch_size=batch_size, verbose=0)

    # extract training time in seconds
    model_info['objectives']['training_time'] = time.time() - start

    # evaluate the model on the test set
    results = model.evaluate(data['test_features'], data['test_labels'], verbose=0)

    # map results to corresponding metrics
    model_info['objectives']['loss'] = results[0]
    model_info['objectives']['accuracy'] = results[1]
    model_info['objectives']['binary_accuracy'] = results[2]
    model_info['objectives']['cosine_similarity'] = results[3]
    model_info['objectives']['false_positives'] = results[4]
    model_info['objectives']['false_negatives'] = results[5]
    model_info['objectives']['true_positives'] = results[6]
    model_info['objectives']['true_negatives'] = results[7]
    model_info['objectives']['mean_squared_error'] = results[8]
    model_info['objectives']['recall'] = results[9]
    model_info['objectives']['precision'] = results[10]
    model_info['objectives']['auc'] = results[11]
    model_info['objectives']['sensitivity'] = results[12]
    model_info['objectives']['specificity'] = results[13]

    # store total layers
    model_info['objectives']['total_layers'] = len(model.layers)

    return model_info

print('model evaluation function defined')

def main():

    # intialize data for model development
    data = data_preprocessing()
    print('data processed')

    # manually design 10 artificial neural network architectures
    ''' outline choices for hyperparameters:
        * number of layers: 2 to 16
        * number of units per layer: 2, 8, 32, 128
        * activation functions: relu, sigmoid, tanh
        * optimizers: adam, sgd, rmsprop
        * epochs: 3 to 12
        * batch size: 16, 32, 64
        * loss function: binary_crossentropy, binary_focal_crossentropy, cosine_similarity
    '''
    ann_models = {
        'model_1': {
            'hyperparameters': {
                'layers': 4, 'units': 128, 'activation': 'relu', 'optimizer': 'adam', 'loss': 'binary_crossentropy',
                'epochs': 3, 'batch_size': 16
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        'model_2': {
            'hyperparameters': {
                'layers': 8, 'units': 32, 'activation': 'relu', 'optimizer': 'sgd', 'loss': 'cosine_similarity',
            'epochs': 4, 'batch_size': 32
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        'model_3': {
            'hyperparameters': {
                'layers': 16, 'units': 8, 'activation': 'relu', 'optimizer': 'rmsprop', 'loss': 'binary_crossentropy',
                'epochs': 6, 'batch_size': 64
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        'model_4': {
            'hyperparameters': {
                'layers': 3, 'units': 32, 'activation': 'sigmoid', 'optimizer': 'adam', 'loss': 'cosine_similarity',
                'epochs': 7, 'batch_size': 16
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        'model_5': {
            'hyperparameters': {
                'layers': 6, 'units': 128, 'activation': 'sigmoid', 'optimizer': 'sgd', 'loss': 'cosine_similarity',
                'epochs': 8, 'batch_size': 32
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        'model_6': {
            'hyperparameters': {
                'layers': 9, 'units': 32, 'activation': 'sigmoid', 'optimizer': 'rmsprop', 'loss': 'binary_crossentropy',
                'epochs': 9, 'batch_size': 64
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        'model_7': {
            'hyperparameters': {
                'layers': 5, 'units': 8, 'activation': 'tanh', 'optimizer': 'adam', 'loss': 'binary_focal_crossentropy',
                'epochs': 10, 'batch_size': 16
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        'model_8': {
            'hyperparameters': {
                'layers': 10, 'units': 128, 'activation': 'tanh', 'optimizer': 'sgd', 'loss': 'binary_focal_crossentropy',
                'epochs': 11, 'batch_size': 32
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        'model_9': {
            'hyperparameters': {
                'layers': 15, 'units': 2, 'activation': 'tanh', 'optimizer': 'rmsprop', 'loss': 'binary_focal_crossentropy',
                'epochs': 12, 'batch_size': 64
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
        # test case from previous simulations
        # 'error_rate', 0.6333333253860474, 'layer_count', 2, 'avg_nodes_per_layer', 2,
        # 'training_time', 0.9758102893829346, 'activation': ['sigmoid', 'sigmoid'],
        # 'optimizer': 'sgd', 'units_per_layer': [1, 3], 'layer_count': 2, 'avg_nodes_per_layer': 2,
        # 'test_accuracy': 0.36666667461395264, loss: binary_crossentropy

        'model_10': {
            'hyperparameters': {
                'layers': 2, 'units': 2, 'activation': 'sigmoid', 'optimizer': 'sgd', 'loss': 'binary_crossentropy',
                'epochs': 5, 'batch_size': 32
            },
            'model_object': None,
            'objectives': {
                'training_time': 0.0, 'total_layers': 0, 'total_nodes': 0, 'loss': 0,
                'accuracy': 0, 'binary_accuracy': 0, 'cosine_similarity': 0,
                'true_positives': 0, 'true_negatives': 0, 'false_positives': 0,
                'false_negatives': 0, 'mean_squared_error': 0, 'recall': 0,
                'precision': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0
            },
            'pareto_optimal': False,
            'validated': False
        },
    }
    print('models designed')

    # build, train and evaluate each model
    class_count = len(np.unique(data['train_labels']))
    feature_count = data['train_features'].shape[1]
    for model_name, model_info in ann_models.items():

        # build model
        model_info = build_neural_network('ann', model_info, class_count, feature_count)

        # evaluate model performance
        model_info = evaluate_model_performance(data, model_info)

        # update model info
        ann_models[model_name] = model_info

    print('metrics populated')

    # store all models and their performance metrics as a csv file (no nested columns)
    records = []
    for model_name, model_info in list(ann_models.items()):
        record = {'model_name': model_name}

        # store each hyperparameter and objective as a separate column
        for hp_name, hp_value in model_info['hyperparameters'].items():
            record[f'hp_{hp_name}'] = hp_value
        for obj_name, obj_value in model_info['objectives'].items():
            record[f'obj_{obj_name}'] = obj_value

        records.append(record)

    print('info tabulated')
    df = pd.DataFrame(records)
    df.to_csv('../../outputs/manual_ann_models_performance.csv', index=False) # remember f1 scores
    print('csv saved')


if __name__ == "__main__":
    main()