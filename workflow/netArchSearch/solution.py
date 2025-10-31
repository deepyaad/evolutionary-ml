import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from keras.metrics import (
    CategoricalAccuracy, BinaryAccuracy
)
from sklearn.metrics import confusion_matrix, roc_auc_score

class Solution:

    def __init__(self, hyperparams, metrics=None, model=None):
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
        self.metrics = metrics if metrics is not None else {}


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
                      loss=loss_function
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
        y_pred_probs = self.model.predict(X, verbose=0)
        if self.hyperparams['class_count'] == 2: 
            y_pred = (y_pred_probs > 0.5).astype(int) # binary classification
        else: 
            y_pred_probs.argmax(axis=1) # multi-class classification

        # core confusion-matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0         # aka specificity
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0        # aka recall
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0         # aka sensitivity
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * precision * false_positive_rate) / (precision + false_positive_rate) if (precision + false_positive_rate) > 0 else 0

        # AUC and accuracy manually for consistency
        auc = roc_auc_score(y, y_pred_probs)
        accuracy = (y_pred.flatten() == y).mean()

        # map results to corresponding metrics
        self.metrics['loss'] = results
        self.metrics['accuracy'] = accuracy
        self.metrics['false_positive_rate'] = false_positive_rate
        self.metrics['false_negative_rate'] = false_negative_rate
        self.metrics['true_positive_rate'] = true_positive_rate
        self.metrics['true_negative_rate'] = true_negative_rate
        self.metrics['f1'] = f1
        self.metrics['precision'] = precision
        self.metrics['auc'] = auc


    def validate_model(self, data):
        """
        purpose: evaluate model on validation data
        params: None
        output: None
        """
        if self.model == None:
            self.develop_model(data)
        
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