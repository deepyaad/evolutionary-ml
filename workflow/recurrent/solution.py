import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from sklearn.metrics import confusion_matrix
from keras import optimizers
import psutil
import os


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
        params: 
            data: Either a dict (legacy format) or LazyDataLoader instance
        output: None
        """
        from mutators import create_layer
        from data_loader import LazyDataLoader

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
        
        feature_selection = self.configuration['feature_selection']
        class_count = self.configuration['class_count']
        labels_inorder = self.configuration['labels_inorder']
        
        epochs = self.configuration['epochs']
        batch_size = self.configuration['batch_size']

        # Handle data loading - support both LazyDataLoader and legacy dict format
        if isinstance(data, LazyDataLoader):
            # Get current feature shape for the selected feature type
            current_feature_shape = data.get_feature_shape(feature_selection)
            # Update configuration if feature shape changed
            if 'feature_shape' not in self.configuration or self.configuration['feature_shape'] != current_feature_shape:
                print(f'Feature shape changed: {self.configuration.get("feature_shape", "unknown")} -> {current_feature_shape}')
                self.configuration['feature_shape'] = current_feature_shape
                # Rebuild layers if feature shape changed
                hidden_layers, layer_names, specifications, outputs = self._rebuild_layers_for_feature_shape(
                    current_feature_shape, hidden_layer_count
                )
                self.configuration['hidden_layers'] = hidden_layers
                self.configuration['layer_names'] = layer_names
                self.configuration['layer_specifications'] = specifications
                self.configuration['neurons_per_layer'] = outputs
            
            # Get data dict for this feature type
            data_dict = data.get_data_dict(feature_selection)
        else:
            # Legacy format - dict with all features
            current_feature_shape = self.configuration['feature_shape']
            data_dict = data

        feature_shape = current_feature_shape
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
        self.model.fit(
            data_dict[f'{feature_selection}_train_features'], data_dict['train_labels'], 
            epochs=epochs, batch_size=batch_size, verbose=0
        )
        print(f'trained model')

        # evaluate model on test data
        print('starting model evaluation')
        self._calculate_metrics(
            data_dict[f'{feature_selection}_test_features'], data_dict['test_labels'], 
            class_count, labels_inorder
        )
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

        # add individual class performance
        for label in labels_inorder:
            self.metrics[f'{label}_performance'] = cm_dict[label]

        # add overall model performance
        self.metrics['loss'] = 999999 if np.isnan(loss) else loss
        self.metrics['precision_macro'] = np.mean([self.metrics[l+'_performance']['precision'] for l in labels_inorder])
        self.metrics['recall_macro'] = np.mean([self.metrics[l+'_performance']['recall_tpr'] for l in labels_inorder])
        self.metrics['f1_macro'] = np.mean([self.metrics[l+'_performance']['f1'] for l in labels_inorder])
        self.metrics['accuracy_macro'] = np.mean([self.metrics[l+'_performance']['accuracy'] for l in labels_inorder])


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
        params: 
            data: Either a dict (legacy format) or LazyDataLoader instance
        output: None
        """
        from data_loader import LazyDataLoader
        
        if self.model == None:
            self.develop_model(data)
        
        feature_selection = self.configuration['feature_selection']
        class_count = self.configuration['class_count']
        
        # Handle data loading
        if isinstance(data, LazyDataLoader):
            data_dict = data.get_data_dict(feature_selection)
            labels_inorder = data.labels_inorder
        else:
            data_dict = data
            labels_inorder = data['labels_inorder']

        # evaluate model on unseen data
        self._calculate_metrics(
            data_dict[f'{feature_selection}_val_features'], data_dict['val_labels'], 
            class_count, labels_inorder
        )
        print(f'validated model')
    
    def _rebuild_layers_for_feature_shape(self, new_feature_shape, hidden_layer_count):
        """
        Rebuild layers when feature shape changes.
        This is needed when switching between feature types with different shapes.
        """
        from mutators import create_layer
        
        hidden_layers, layer_names, specifications, outputs = [], [], [], []
        input_size = new_feature_shape
        
        # Try to preserve layer types from existing configuration if possible
        existing_layer_names = self.configuration.get('layer_names', [])
        existing_specs = self.configuration.get('layer_specifications', [])
        
        for i in range(hidden_layer_count):
            is_last = (i == hidden_layer_count - 1)
            
            # Try to reuse existing layer type if available
            if i < len(existing_layer_names):
                layer_type = existing_layer_names[i]
                specs = existing_specs[i] if i < len(existing_specs) else None
                
                if is_last:
                    valid_last_layers = ['LSTM', 'SimpleRNN', 'GRU','GlobalAveragePooling1D', 'GlobalMaxPooling1D']
                    if layer_type in valid_last_layers:
                        layer, name, spec, output_size = create_layer(
                            input_size, [layer_type], specs=specs, last_layer=True
                        )
                    else:
                        # Fallback to random valid last layer
                        layer, name, spec, output_size = create_layer(
                            input_size, valid_last_layers, last_layer=True
                        )
                else:
                    valid_middle_layers = ['LSTM', 'SimpleRNN', 'GRU']
                    if layer_type in valid_middle_layers:
                        layer, name, spec, output_size = create_layer(
                            input_size, [layer_type], specs=specs, last_layer=False
                        )
                    else:
                        # Fallback to random valid middle layer
                        layer, name, spec, output_size = create_layer(
                            input_size, valid_middle_layers, last_layer=False
                        )
            else:
                # Create new layer if we don't have existing config
                if is_last:
                    valid_last_layers = ['LSTM', 'SimpleRNN', 'GRU','GlobalAveragePooling1D', 'GlobalMaxPooling1D']
                    layer, name, spec, output_size = create_layer(
                        input_size, valid_last_layers, last_layer=True
                    )
                else:
                    layer, name, spec, output_size = create_layer(
                        input_size, ['LSTM', 'SimpleRNN', 'GRU'], last_layer=False
                    )
            
            hidden_layers.append(layer)
            layer_names.append(name)
            specifications.append(spec)
            outputs.append(output_size)
            input_size = output_size
        
        return hidden_layers, layer_names, specifications, outputs


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
        layer_order_str = ' → '.join([name for name in self.configuration['layer_names']])
        loss_func_str = self.configuration['loss_function']
        optimizer_str = self.configuration['optimizer']
        epochs_str = self.configuration['epochs']
        batch_size_str = self.configuration['batch_size']
        solution_str = f"I → {layer_order_str} → O with {loss_func_str} loss • {optimizer_str} optimizer • {epochs_str} epochs • {batch_size_str} batch size"

        return solution_str
    


