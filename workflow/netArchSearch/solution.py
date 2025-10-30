import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from keras.metrics import (
    Accuracy, BinaryAccuracy, CosineSimilarity,
    TruePositives, TrueNegatives,
    MeanSquaredError,
    Recall, Precision, AUC,
    FalsePositives, FalseNegatives,
    SensitivityAtSpecificity, SpecificityAtSensitivity
)

class Solution:

    def __init__(self, hyperparams, model=None, metrics=None):
        """
        purpose: initialize solution object
        params:
          hyperparams: dictionary of hyperparameters
          data: dictionary of data
          model: trained model object
          metrics: dictionary of performance metrics
        output:
        """
        # constructor inputs to state variables
        self.hyperparams = hyperparams
        self.model = model
        self.metrics = {}


    def develop_model(self, data):
        """
        purpose: compile, train and test model object based on hyperparameters
        params: None
        output: None
        """

        # get hyperparameter specifications
        activation = self.hyperparams['activation_per_hidden_layer']
        optimizer = self.hyperparams['optimizer']
        loss_function = self.hyperparams['loss_function']
        units_per_hidden_layer = self.hyperparams['units_per_hidden_layer']
        feature_count = self.hyperparams['feature_count']
        class_count = self.hyperparams['class_count']
        hidden_layer_count = self.hyperparams['hidden_layer_count']
        epochs = self.hyperparams['epochs']
        batch_size = self.hyperparams['batch_size']


        # intialize the model
        self.model = keras.Sequential()

        # initialize input layer
        self.model.add(layers.InputLayer(shape=(feature_count,)))

        # add hidden layers
        for i in range(hidden_layer_count):
            self.model.add(layers.Dense(units_per_hidden_layer[i], activation=activation[i]))

        # add output layer
        if class_count == 2: 
            self.model.add(layers.Dense(1, activation='sigmoid')) # binary classification
        else: 
            self.model.add(layers.Dense(class_count, activation='softmax')) # multi-class classification

        # compile the model
        self.model.compile(optimizer=optimizer,
                      loss=loss_function,
            metrics=[
                Accuracy(), BinaryAccuracy(), CosineSimilarity(),
                FalsePositives(), FalseNegatives(), TruePositives(), TrueNegatives(),
                MeanSquaredError(), Recall(), Precision(), AUC(),
                SensitivityAtSpecificity(specificity=0.9), SpecificityAtSensitivity(sensitivity=0.9)
            ]
        )

        # get the model size metrics
        self.metrics['total_layers'] = hidden_layer_count + 2 # including input and output layers
        self.metrics['total_nodes'] = sum(units_per_hidden_layer) + feature_count + class_count # including input and output nodes

        # intialize timer
        start = time.time()

        # train the model
        self.model.fit(data['train_features'], data['train_labels'], epochs=epochs, batch_size=batch_size, verbose=0)

        # extract training metrics
        self.metrics['training_time'] = time.time() - start

        # evaluate model on test data
        self._calculate_metrics(data['test_features'], data['test_labels'])



    def _calculate_metrics(self, X, y):
        """Helper function to run predictions and calculate all metrics while"""

        # evaluate the model on the test set
        results = self.model.evaluate(X, y, verbose=0)

        # map results to corresponding metrics
        self.metrics['loss'] = results[0]
        self.metrics['accuracy'] = results[1]
        self.metrics['binary_accuracy'] = results[2]
        self.metrics['cosine_similarity'] = results[3]
        self.metrics['false_positive_rate'] = results[4] / len(y)
        self.metrics['false_negative_rate'] = results[5] / len(y)
        self.metrics['true_positive_rate'] = results[6] / len(y)
        self.metrics['true_negative_rate'] = results[7] / len(y)
        self.metrics['mean_squared_error'] = results[8]
        self.metrics['recall'] = results[9]
        self.metrics['precision'] = results[10]
        self.metrics['auc'] = results[11]
        self.metrics['sensitivity'] = results[12]
        self.metrics['specificity'] = results[13]

    def validate_model(self, data):
        """
        purpose: evaluate model on validation data
        params: None
        output: None
        """

        # evaluate model on unseen data
        self._calculate_metrics(data['val_features'], data['val_labels'])


    def to_dict(self):
      """
      purpose: convert solution object to dictionary without model object for easy archiving
      params: None
      output: dictionary
      """

      return {
            'hyperparams': self.hyperparams,
            'metrics': self.metrics
        }



    def __repr__(self):
        return f"Solution(hyperparams={self.hyperparams}, metrics={self.metrics})"