import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from sklearn.metrics import confusion_matrix
from keras import optimizers
import psutil
import os
import uuid


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
        from mutators import create_layer

        print('starting model development')
        
        # intialize
        start_time = time.time()
        pid = os.getpid()
        start_resource_usage = self._get_current_resource_usage(pid)

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

        # train the model
        print('starting model training')
        self.model.fit(data['train_features'], data['train_labels'], epochs=epochs, batch_size=batch_size, verbose=0)
        print(f'trained model')

        # evaluate model on test data
        print('starting model evaluation')
        self._calculate_metrics(data['test_features'], data['test_labels'], class_count, data['labels_inorder'])
        print(f'evaluated model')

        # calculate remaining metrics
        development_duration = time.time() - start_time
        end_resource_usage = self._get_current_resource_usage(pid)

        self.metrics['cpu_util_percent'] = end_resource_usage['cpu_percent'] - start_resource_usage['cpu_percent']
        self.metrics['ram_usage_mb'] = end_resource_usage['ram_mb'] - start_resource_usage['ram_mb']
        self.metrics['development_time'] = development_duration

        # print results
        ram_mb = self.metrics['ram_usage_mb']
        print(f'model development took {development_duration} secs') # and used {ram_mb} MB of RAM')

    # TODO: fix memory usage calculation
    def _get_current_resource_usage(self, pid):
        process = psutil.Process(pid)
        cpu_percent = process.cpu_percent(interval=0.1)
        ram_usage_mb = process.memory_info().rss / (1024 * 1024)
        return {'cpu_percent': cpu_percent, 'ram_mb': ram_usage_mb}


    def _calculate_metrics(self, X, y, class_count, labels_inorder):

        # intialize timer
        start = time.time()

        # calculate latency
        single_pred = self.model.predict(X[:1], verbose=0)
        latency = time.time() - start
        self.metrics['latency'] = latency

        # calculate throughput
        batch_size = 32
        batch_times, batch_sizes = [], []
        num_samples = len(X)
        remainder = num_samples % batch_size
        num_samples -= remainder
        for i in range(0, num_samples, batch_size):
            batch = X[i:i+batch_size]
            t0 = time.time()
            _ = self.model.predict(batch, verbose=0)
            t1 = time.time()
            batch_times.append(t1 - t0)
            batch_sizes.append(len(batch))
        self.metrics['throughput'] = sum(batch_sizes)/sum(batch_times)


        # evaluate the model on the test set
        loss = self.model.evaluate(X, y, verbose=0)
        y_pred_probs = self.model.predict(X, verbose=0)
        y_pred = y_pred_probs.argmax(axis=1)                                                                                # multi-class classification


        # core confusion-matrix derived metrics
        cm = confusion_matrix(y.argmax(axis=1), y_pred)
        cm_dict = {}
        total = cm.sum()
        for i in range(class_count):
            language = labels_inorder[i]
            tp = cm[i][i]                                                                                                   # you predicted me and i was there
            fp = cm[:, i].sum() - tp                                                                                        # you predicted me and i wasn't there
            fn = cm[i].sum() - tp                                                                                           # you didn't predict me and i was there
            tn = total - (tp + fp + fn)                                                                                     # you didn't predict me and i wasn't there

            accuracy = (tp + tn) / total                                                                                    # use as a rough indicator of model training progress
            recall_tpr = tp / (tp + fn) if (tp + fn) != 0 else 0                                                            # use when fn (type II) are more expensive than fp (type I) i.e. medicine
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0                                                                   # use when fp (type I) are more expensive than fn (type II)
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0                                                             # use when it's very important for positive predictions to be accurate
            f1 = 2 * (precision * recall_tpr) / (precision + recall_tpr) if (precision + recall_tpr) != 0 else 0
            

            # map results to metrics
            cm_dict[language] = {
                'true_pos': tp, 'true_neg': tn, 'false_pos': fp, 'false_neg': fn,
                'accuracy': accuracy, 'recall': recall_tpr, 'false_pos_rate': fpr,
                'precision': precision, "f1": f1
            }

        # add individual class performance
        for language in labels_inorder:
            for key, value in cm_dict[language].items():
                self.metrics[f'{language}_{key}'] = value

        # add overall model performance
        self.metrics['loss'] = 999999 if np.isnan(loss) else loss
        self.metrics['precision_macro'] = np.mean([self.metrics[l+'_precision'] for l in labels_inorder])
        self.metrics['recall_macro'] = np.mean([self.metrics[l+'_recall'] for l in labels_inorder])
        self.metrics['f1_macro'] = np.mean([self.metrics[l+'_f1'] for l in labels_inorder])
        self.metrics['accuracy_macro'] = np.mean([self.metrics[l+'_accuracy'] for l in labels_inorder])


    def _calculate_flops(self):
        '''
        purpose: calculate FLOPs per forward pass
        TODO: literature for FLOPs formula for each layer type and implementation in Keras
        params: None
        output: None
        '''
        # calculate FLOPs per forward pass
        total_flops = 0
        features = self.configuration['feature_shape'][1]
        timesteps = self.configuration['feature_shape'][0]
        for layer in self.model.layers:
            pass
            
            # input layer calculations (InputLayer)

            # recurrent layers calculations (LSTM, GRU, SimpleRNN, Bidirectional, ConvLSTM1D)

            # convolutional layers calculations (Conv1D, Conv1DTranspose, SeparableConv1D, ConvLSTM1D)

            # dense layers calculations (Dense)

            # pooling layers calculations (MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D)

            # dropout layers calculations (Dropout)

            # normalization layers calculations (Normalization, SpectralNormalization)


    def validate_model(self, data):
        """
        purpose: evaluate model on validation data
        params: None
        output: None
        """
        if self.model == None:
            self.develop_model(data)
        
        # evaluate model on unseen data
        self._calculate_metrics(data['val_features'], data['val_labels'], 
                               self.configuration['class_count'], data['labels_inorder'])


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
            elif isinstance(obj, uuid.UUID):
                return str(obj)
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
        layer_order_str = ' → '.join([name for name in self.configuration['layer_names']])
        loss_func_str = self.configuration['loss_function']
        optimizer_str = self.configuration['optimizer']
        epochs_str = self.configuration['epochs']
        batch_size_str = self.configuration['batch_size']
        solution_str = f"I → {layer_order_str} → O with {loss_func_str} loss • {optimizer_str} optimizer • {epochs_str} epochs • {batch_size_str} batch size"

        return solution_str
    


