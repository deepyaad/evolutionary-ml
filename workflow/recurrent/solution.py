import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from sklearn.metrics import confusion_matrix
from keras import optimizers


class Solution:

    def __init__(self, configuration, metrics=None, model=None):
        """
        purpose: initialize solution object
        params:
          configuration: dictionary of configuration for architecture and hyperparameters
          data: dictionary of data
          model: trained model object
          metrics: dictionary of performance metrics
        output:
        """
        # constructor inputs to state variables
        self.configuration = configuration
        self.model = model
        self.metrics = metrics if metrics is not None else {}


    def develop_model(self, data):
        """
        purpose: compile, train and test model object based on hyperparameters
        params: None
        output: None
        """
        from agents import create_layer

        print('starting model developmet')
        # intialize timer
        start = time.time()

        # get configuration specifications
        hidden_layer_count = self.configuration['hidden_layer_count']
        hidden_layers = self.configuration['hidden_layers']

        loss_function = self.configuration['loss_function']
        optimizer = self.configuration['optimizer']
        
        feature_shape = self.configuration['feature_shape']
        class_count = self.configuration['class_count']
        
        epochs = self.configuration['epochs']
        batch_size = self.configuration['batch_size']

        print('loaded configuration specifications')
  

        # intialize the model
        self.model = keras.Sequential()

        # initialize input layer
        self.model.add(layers.InputLayer(shape=(feature_shape)))
        print('initialized input layer')
        

        # add hidden layers
        print(f'adding {hidden_layer_count} hidden layers')
        for layer in hidden_layers:
            self.model.add(layer)
        print(f'finished hidden layers')
            

        # add output layer for multi-class classification
        self.model.add(layers.Dense(class_count, activation='softmax')) 
        print(f'added output layer')

        # compile the model
        optimizer_dict = {
            'adam': optimizers.Adam(), 'sgd': optimizers.SGD(), 'rmsprop':  optimizers.RMSprop(), 
            'adamw': optimizers.AdamW(), 'adadelta':  optimizers.Adadelta(), 'adagrad': optimizers.Adagrad(), 
            'adamax': optimizers.Adamax(), 'adafactor': optimizers.Adafactor(), 'nadam': optimizers.Nadam(), 
            'ftrl': optimizers.Ftrl(), 'lion': optimizers.Lion(), 'lamb': optimizers.Lamb()
        }
        optimizer_func = optimizer_dict[optimizer]
        self.model.compile(optimizer=optimizer_func,
                      loss=loss_function
        )
        print(f'compiled model')

        # get the model size metrics
        self.metrics['total_layers'] = hidden_layer_count + 2 # including input and output layers
        # self.metrics['total_nodes'] = sum(units_per_hidden_layer) + feature_count + class_count # including input and output nodes

        # train the model
        print('starting model training')
        self.model.fit(data['train_features'], data['train_labels'], epochs=epochs, batch_size=batch_size, verbose=0)
        print(f'trained model')

        # evaluate model on test data
        print('starting model evaluation')
        self._calculate_metrics(data['test_features'], data['test_labels'], class_count, data['labels_inorder'])
        print(f'evaluated model')

        # end timer
        training_duration = time.time() - start
        self.metrics['development_time'] = training_duration
        print(f'model training took {training_duration} secs ')


    def _build_model_architecture(self):
        """
        nuild model architecture without training for weight transfer
        """
        from agents import create_layer
        
        # get configuration specifications
        hidden_layer_count = self.configuration['hidden_layer_count']
        hidden_layers = self.configuration['hidden_layers']
        feature_shape = self.configuration['feature_shape']
        class_count = self.configuration['class_count']
        
        # intialize the model
        self.model = keras.Sequential()
        
        # initialize input layer
        self.model.add(layers.InputLayer(shape=(feature_shape)))
        
        # add hidden layers
        for layer in hidden_layers:
            self.model.add(layer)
        
        # add output layer
        self.model.add(layers.Dense(class_count, activation='softmax'))


    def _compile_and_finetune(self, data):
        """
        Compile and train with reduced epochs since we have transferred weights
        """
        # get configuration specifications
        loss_function = self.configuration['loss_function']
        optimizer = self.configuration['optimizer']
        epochs = self.configuration['epochs']
        batch_size = self.configuration['batch_size']
        
        # compile the model
        optimizer_dict = {
            'adam': optimizers.Adam(), 'sgd': optimizers.SGD(), 'rmsprop': optimizers.RMSprop(), 
            'adamw': optimizers.AdamW(), 'adadelta': optimizers.Adadelta(), 'adagrad': optimizers.Adagrad(), 
            'adamax': optimizers.Adamax(), 'adafactor': optimizers.Adafactor(), 'nadam': optimizers.Nadam(), 
            'ftrl': optimizers.Ftrl(), 'lion': optimizers.Lion(), 'lamb': optimizers.Lamb()
        }
        optimizer_func = optimizer_dict[optimizer]
        
        self.model.compile(optimizer=optimizer_func, loss=loss_function)
        
        # use fewer epochs for fine-tuning since we start with good weights
        fine_tune_epochs = max(3, epochs // 2)
        print(f"Fine-tuning with {fine_tune_epochs} epochs (reduced from {epochs})")
        
        # train the model with reduced epochs
        start = time.time()
        self.model.fit(data['train_features'], data['train_labels'], 
                      epochs=fine_tune_epochs, batch_size=batch_size, verbose=0)
        
        # evaluate model
        self._calculate_metrics(data['test_features'], data['test_labels'], 
                              self.configuration['class_count'], data.get('labels_inorder', []))
        
        training_duration = time.time() - start
        self.metrics['development_time'] = training_duration
        print(f"Fine-tuning took {training_duration} secs")


    def _calculate_metrics(self, X, y, class_count, labels_inorder):
        """Helper function to run predictions and calculate all metrics while"""

        # evaluate the model on the test set
        results = self.model.evaluate(X, y, verbose=0)
        y_pred_probs = self.model.predict(X, verbose=0)
        y_pred = y_pred_probs.argmax(axis=1) # multi-class classification


        # core confusion-matrix derived metrics
        cm = confusion_matrix(y.argmax(axis=1), y_pred)
        cm_dict = {}
        total = cm.sum()
        for i in range(class_count):
            label = labels_inorder[i]
            tp = cm[i][i]                                                                                                   # you predicted me and i was there
            fp = cm[:, i].sum() - tp                                                                                        # you predicted me and i wasn't there
            fn = cm[i].sum() - tp                                                                                           # you didn't predict me and i was there
            tn = total - (tp + fp + fn)                                                                                     # you didn't predict me and i wasn't there

            accuracy = (tp + tn) / total                                                                                    # use as a rough indicator of model training progress
            recall_tpr = tp / (tp + fn) if (tp + fn) != 0 else 0                                                            # use when fn (type II) are more expensive than fp (type I) i.e. medicine
            fpr = fp / (fp + tn)                                                                                            # use when fp (type I) are more expensive than fn (type II)
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0                                                             # use when it's very important for positive predictions to be accurate
            f1 = 2 * (precision * recall_tpr) / (precision + recall_tpr) if (precision + recall_tpr) != 0 else 0
            

            # map results to metrics
            cm_dict[label] = {
                'true_pos': tp, 'true_neg': tn, 'false_pos': fp, 'false_neg': fn,
                'accuracy': accuracy, 'recall_tpr': recall_tpr, 'false_pos_rate': fpr,
                'precision': precision, "f1": f1
            }

        # join all classes into one dictionary
        self.metrics = cm_dict

        # add overall model performance
        # overall model metrics
        self.metrics['model'] = {
            'loss': results,
            'precision_macro': np.mean([self.metrics[l]['precision'] for l in labels_inorder]),
            'recall_macro': np.mean([self.metrics[l]['recall_tpr'] for l in labels_inorder]),
            'f1_macro': np.mean([self.metrics[l]['f1'] for l in labels_inorder]),
            'accuracy_macro': np.mean([self.metrics[l]['accuracy'] for l in labels_inorder])
        }
        self.metrics['model']['loss'] = 999999 if np.isnan(results) else results
        print('LOSS: ', results)


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
        def safe_convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: safe_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [safe_convert(v) for v in obj]
            else:
                return obj
        
        json_serializable_configs = {k: safe_convert(v) for k, v in self.configuration.items() if k != 'hidden_layers'}
        json_serializable_metrics = {k: safe_convert(v) for k, v in self.metrics.items()}
        
        return {'configuration': json_serializable_configs, 'metrics': json_serializable_metrics}


    def __repr__(self):
        return f"Solution(configuration={self.configuration}"
    


