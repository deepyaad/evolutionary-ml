from solution import Solution
import random as rnd
from profiler import profile
import random as rnd
from tensorflow.keras import layers
from pprint import pprint as prtty
import uuid


def create_layer(input_size, layer_types=[], last_layer=False, specs=None):

  # intialize variables
  activations = [
    'celu', 'elu', 'gelu', 'hard_sigmoid', 'hard_shrink', 'hard_tanh', 'hard_silu', 
    'leaky_relu', 'linear', 'mish', 'relu', 'selu', 'silu', 
    'sigmoid', 'softmax', 'softplus', 'softsign', 'soft_shrink', 'swish', 'tanh',
    'tanh_shrink'
  ]

  specifications = None
           

  # select layer type
  if len(layer_types) > 0:
    layer = rnd.choice(layer_types)

  # TODO: add TimeDistributed Layers
  else:
    if last_layer:
      # When last_layer=True, only choose layers that can remove temporal dimension
      layer = rnd.choice([
        'LSTM', 'SimpleRNN', 'GRU', 
        'GlobalAveragePooling1D', 'GlobalMaxPooling1D'
      ])
    else:
      layer = rnd.choice([
        'Dropout', 'Dense',                                                                              # Dense Layers          
        'Normalization', 'SpectralNormalization',                                                        # Normalization Layers
        'SeparableConv1D', 'Conv1D', 'Conv1DTranspose',                                                  # Convolution Layers         
        'MaxPooling1D', 'AveragePooling1D',                                                              # Pooling Layers
        'LSTM', 'SimpleRNN', 'GRU', 'Bidirectional', 'ConvLSTM1D'                                        # Recurrent Layers
      ])

  # ensure last temporal layer removes time steps
  if last_layer == True:
    return_sequences = False
  else:
    return_sequences = True   

  # determine RNN units - use specs if available, otherwise use input_size (if int) or random
  if specs is not None and 'units' in specs:
    # use units from specs (when rebuilding existing layers)
    rnn_units = specs['units']
  elif last_layer and layer in ['LSTM', 'SimpleRNN', 'GRU']:
    # if input_size is an integer, use it; if tuple (feature_shape), use random
    if isinstance(input_size, tuple):
      # input_size is feature_shape tuple (469, 96) - can't use as units, use random
      rnn_units = rnd.randint(10, 60)
    else:
      # input_size is integer (from previous layer's output)
      rnn_units = input_size
  else:
    rnn_units = rnd.randint(10, 60)


  # regular densely-connected NN layer
  if layer == 'Dense':
    units = rnd.randint(6, 32)
    if specs is None: 
      activation = rnd.choice(activations)
    else:
      activation = specs['activation']
    
    specifications = {}
    specifications['units'] = units
    specifications['activation'] = activation
    tf_layer = layers.Dense(units, activation=activation)
    output_size = units
      


  # preprocessing layer that normalizes continuous features
  elif layer == 'Normalization':
    tf_layer = layers.Normalization()
    output_size = input_size

  # Performs spectral normalization on the weights of a target layer
  elif layer == 'SpectralNormalization':
    # SpectralNormalization only works with layers that have a single 'kernel' attribute
    layer_with_kernel, name, specs, output_size = create_layer(input_size, ['Dense', 'Conv1D'])
    if layer_with_kernel is None:
        raise ValueError("SpectralNormalization wrapper requires a valid layer_with_kernel")

    specifications = {}
    specifications['target_layer'] = name
    specifications['return_sequences'] = return_sequences
    # Copy the actual specs values, not the key names
    for k in specs.keys():
      specifications[k] = specs[k]

    tf_layer = layers.SpectralNormalization(layer_with_kernel)
  

  # 1D convolution layer (e.g. temporal convolution)
  elif layer == 'Conv1D':
    filters = rnd.randint(8, 32)
    if specs is None: 
      kernel_size = rnd.randint(3, 24)
      activation = rnd.choice(activations)
    else:
      kernel_size = specs['kernel_size']
      activation = specs['activation']

    specifications = {}
    specifications['filters'] = filters
    specifications['kernel_size'] = kernel_size
    specifications['activation'] = activation
    tf_layer = layers.Conv1D(filters, kernel_size, activation=activation)
    output_size = filters


  # 1D separable convolution layer
  elif layer == 'SeparableConv1D':
    filters = rnd.randint(8, 32)
    if specs is None: 
      kernel_size = rnd.randint(3, 24)
      activation = rnd.choice(activations)
    else:
      kernel_size = specs['kernel_size']
      activation = specs['activation']

    specifications = {}
    specifications['filters'] = filters
    specifications['kernel_size'] = kernel_size
    specifications['activation'] = activation                                       
    tf_layer = layers.SeparableConv1D(filters, kernel_size, activation=activation)
    output_size = filters


  # 1D transposed convolution layer
  elif layer == 'Conv1DTranspose':   
    filters = rnd.randint(8, 32)                                             
    if specs is None: 
      kernel_size = rnd.randint(3, 24)
      activation = rnd.choice(activations)
    else:
      kernel_size = specs['kernel_size']
      activation = specs['activation']

    specifications = {}
    specifications['filters'] = filters
    specifications['kernel_size'] = kernel_size
    specifications['activation'] = activation                                  
    tf_layer = layers.SeparableConv1D(filters, kernel_size, activation=activation)
    output_size = filters

  # 1D convolution LSTM (input and recurrent transformations are convolutional)
  elif layer == 'ConvLSTM1D':
    filters = rnd.randint(8, 32)
    if specs is None: 
      kernel_size = rnd.randint(3, 24)
      activation = rnd.choice(activations)
      recurrent_activation = rnd.choice(activations)
    else:
      kernel_size = specs.get('kernel_size', rnd.randint(3, 24))
      activation = specs.get('activation', rnd.choice(activations))
      recurrent_activation = specs.get('recurrent_activation', rnd.choice(activations))

    specifications = {}
    specifications['filters'] = filters
    specifications['kernel_size'] = kernel_size
    specifications['activation'] = activation
    tf_layer = layers.Conv1D(filters, kernel_size, activation=activation)
    output_size = filters
  
  # Long Short-Term Memory layer - Hochreiter 1997
  elif layer == 'LSTM':
    units = rnn_units
    if specs is None: 
      recurrent_dropout = 0
      activation = rnd.choice(activations)
      recurrent_activation = rnd.choice(activations)   
    else:
      recurrent_dropout = specs.get('recurrent_dropout', 0)
      activation = specs.get('activation', rnd.choice(activations))
      recurrent_activation = specs.get('recurrent_activation', rnd.choice(activations))

    specifications = {}
    specifications['units'] = units
    specifications['activation'] = activation
    specifications['recurrent_activation'] = recurrent_activation
    specifications['recurrent_dropout'] = recurrent_dropout
    specifications['return_sequences'] = return_sequences

    tf_layer = layers.LSTM(
      units, activation = activation,
      recurrent_activation = recurrent_activation,
      recurrent_dropout = recurrent_dropout,
      return_sequences = return_sequences
    )
    output_size = units

  # Gated Recurrent Unit - Cho et al. 2014
  elif layer == 'GRU':
    units = rnn_units
    if specs is None: 
      recurrent_dropout = 0
      activation = rnd.choice(activations)
      recurrent_activation = rnd.choice(activations)
    else:
      recurrent_dropout = specs.get('recurrent_dropout', 0)
      activation = specs.get('activation', rnd.choice(activations))
      recurrent_activation = specs.get('recurrent_activation', rnd.choice(activations))

    specifications = {}
    specifications['units'] = units
    specifications['activation'] = activation
    specifications['recurrent_activation'] = recurrent_activation
    specifications['recurrent_dropout'] = recurrent_dropout
    specifications['return_sequences'] = return_sequences

    tf_layer = layers.GRU(
      units, activation = activation,
      recurrent_activation = recurrent_activation,
      recurrent_dropout = recurrent_dropout,
      return_sequences = return_sequences
    )
    output_size = units
      

  # Fully-connected RNN where the output is to be fed back as the new input
  elif layer == 'SimpleRNN':
    units = rnn_units
    specifications = {}
    specifications['units'] = units
    specifications['return_sequences'] = return_sequences

    tf_layer = layers.SimpleRNN(units, return_sequences = return_sequences)
    output_size = units
  

  # Bidirectional wrapper for RNNs
  elif layer == 'Bidirectional':
    rnn_layer, name, specs, output_size = create_layer(input_size, ['LSTM', 'GRU'])
    if rnn_layer is None:
        raise ValueError("Bidirectional wrapper requires a valid rnn_layer")
    
    merge_mode = rnd.choice(["sum", "mul", "concat", "ave"])

    specifications = {}
    specifications['merge_mode'] = merge_mode
    specifications['rnn_layer'] = name 
    specifications['return_sequences'] = return_sequences
    # Copy the actual specs values, not the key names
    for k in specs.keys():
      specifications[k] = specs[k]
       
    tf_layer = layers.Bidirectional(rnn_layer, merge_mode = merge_mode)


    # Max pooling operation for 1D temporal data
  elif layer == 'MaxPooling1D':
    if specs is None: 
      pool_size = rnd.randint(1, 5)
      strides = rnd.randint(1, pool_size)
      padding = rnd.choice(['valid', 'same'])
    else:
      pool_size = specs['pool_size']
      strides = specs['strides']
      padding = specs['padding']

    specifications = {}
    specifications['pool_size'] = pool_size
    specifications['strides'] = strides
    specifications['padding'] = padding
    tf_layer = layers.MaxPooling1D(pool_size, strides, padding)
    output_size = input_size

  
  # Average pooling for temporal data
  elif layer == 'AveragePooling1D':
    if specs is None: 
      pool_size = rnd.randint(1, 5)
      strides = rnd.randint(1, pool_size)
      padding = rnd.choice(['valid', 'same'])
    else:
      pool_size = specs['pool_size']
      strides = specs['strides']
      padding = specs['padding']

    specifications = {}
    specifications['pool_size'] = pool_size
    specifications['strides'] = strides
    specifications['padding'] = padding
    tf_layer = layers.AveragePooling1D(pool_size, strides, padding)
    output_size = input_size
  

  # Global average pooling operation for temporal data
  elif layer == 'GlobalAveragePooling1D':
    tf_layer = layers.GlobalAveragePooling1D()
    output_size = input_size


  # Global max pooling operation for temporal data
  elif layer == 'GlobalMaxPooling1D':
    tf_layer = layers.GlobalMaxPooling1D()
    output_size = input_size
  

  # Applies dropout to the input
  elif layer == 'Dropout':
    if specs is None: 
      rate = rnd.random() / 3 
    else:
      rate = specs['rate']
    
    specifications = {}
    specifications['rate'] = rate

    tf_layer = layers.Dropout(rate)
    output_size = input_size


  else:
    raise ValueError(f"Layer '{layer}' not implemented in create_layer.")

  return tf_layer, layer, specifications, output_size


def config_update(
    sol, data, mutator, hidden_layer_count=None, hidden_layers=None, layer_names=None, specifications=None, outputs=None, 
    loss_func=None, optimizer=None, epochs=None, batch_size=None, id=None, parent_id=None
  ):
    
  new_configs = {

    # architecture
    'hidden_layer_count': hidden_layer_count if hidden_layer_count else sol.configuration['hidden_layer_count'], 
    'hidden_layers': hidden_layers if hidden_layers else sol.configuration['hidden_layers'],
    'layer_names': layer_names if layer_names else sol.configuration['layer_names'],
    'layer_specifications': specifications if specifications else sol.configuration['layer_specifications'],
    'neurons_per_layer': outputs if outputs else sol.configuration['neurons_per_layer'],

    # hyperparameters
    'loss_function': loss_func if loss_func else sol.configuration['loss_function'],
    'optimizer': optimizer if optimizer else sol.configuration['optimizer'],
    'epochs': epochs if epochs else sol.configuration['epochs'],
    'batch_size': batch_size if batch_size else sol.configuration['batch_size'],

    # input data
    'input_size': sol.configuration['input_size'],
    'output_size': sol.configuration['output_size'],
    'feature_shape': sol.configuration['feature_shape'],
    'class_count': sol.configuration['class_count'],
    'labels_inorder': sol.configuration['labels_inorder'],

    # genetic information
    'id': id if id else uuid.uuid4(),
    'parent_id': parent_id if parent_id else sol.configuration['id'],
    'mutator': mutator if mutator else sol.configuration['mutator']
  }


  # generate new solution and train from scratch
  new_sol = Solution(new_configs)
  new_sol.develop_model(data)

  return new_sol


@profile
def add_layer(sol, data):
  '''
  purpose: agent to add a layer to the neural network
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  print(f'adding a layer to...')
  prtty(sol)
  print('--------------------------------')

  # Get current configuration
  hidden_layer_count = sol.configuration['hidden_layer_count']
  neurons_per_layer = sol.configuration['neurons_per_layer']
  feature_shape = sol.configuration['feature_shape']
  
  # Randomly select where to insert the new layer (can be after any layer, including last)
  # -1 means before first layer, last_idx means after last layer
  insert_after_idx = rnd.randint(-1, hidden_layer_count - 1)
  
  # Determine input size for new layer
  if insert_after_idx == -1:
    # Inserting before first layer - use feature_shape
    prev_layer_size = feature_shape
  else:
    # Inserting after layer at insert_after_idx
    prev_layer_size = neurons_per_layer[insert_after_idx]

  # Determine if this will be the last layer after insertion
  will_be_last = (insert_after_idx == hidden_layer_count - 1)
  
  # create new layer
  new_layer, name, specs, output_size = create_layer(prev_layer_size, last_layer=will_be_last)

  # add new layer information to architecture configuration
  hidden_layer_count = hidden_layer_count + 1
  layer_names = sol.configuration['layer_names'].copy()
  specifications = sol.configuration['layer_specifications'].copy()
  outputs = sol.configuration['neurons_per_layer'].copy()
  
  # Insert new layer at the correct position (after insert_after_idx)
  insert_position = insert_after_idx + 1
  layer_names.insert(insert_position, name)
  specifications.insert(insert_position, specs)

  # Extract integer from output_size (handle both tuple and int)
  if isinstance(output_size, tuple):
      output_int = output_size[1] if len(output_size) > 1 else output_size[0]  # Get feature dimension from tuple
  elif isinstance(output_size, int):
      output_int = output_size  # Already an integer
  else:
      raise ValueError(f"Unexpected output_size type: {type(output_size)}")
  outputs.insert(insert_position, output_int)

  # Rebuild all layers from specifications (don't reuse old layer objects!)
  hidden_layers = []
  input_size = feature_shape
  temporal_removing_layers = ['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D']
  
  for i in range(hidden_layer_count):
    is_last = (i == hidden_layer_count - 1)
    if is_last:
      # Check if the last layer can remove temporal dimension
      last_layer_type = layer_names[i]
      if last_layer_type not in temporal_removing_layers:
        # Replace with a temporal-removing layer
        replacement_type = rnd.choice(['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D'])
        layer, name, spec, out_size = create_layer(input_size, [replacement_type], specs=None, last_layer=True)
        # Update the layer name and specs
        layer_names[i] = name
        specifications[i] = spec
        if isinstance(out_size, int):
          outputs[i] = out_size
        elif isinstance(out_size, tuple):
          outputs[i] = out_size[1] if len(out_size) > 1 else out_size[0]
        else:
          outputs[i] = 32  # fallback
      else:
        layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
    else:
      layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
    hidden_layers.append(layer)
    input_size = out_size

  # create new Solution with new layer information
  new_sol = config_update(
    sol, data, mutator='add_layer', hidden_layer_count=hidden_layer_count, hidden_layers=hidden_layers, 
    layer_names=layer_names, specifications=specifications, outputs=outputs
  )

  return new_sol


@profile
def remove_layer(sol, data):
  '''
  purpose: agent to remove a layer to the neural network
  params:
    sol: Solution object containing configuration, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  if sol.configuration['hidden_layer_count'] <= 1:
    print(f'cannot remove layer - only {sol.configuration["hidden_layer_count"]} layers remaining')

    # randomly select different mutator
    print('selecting different mutator')
    mutator = rnd.choice([
      add_layer, swap_layers, grow_layer, shrink_layer, change_activation, 
      change_optimizer, change_loss_func, change_batch_size,
      change_epochs
    ])
    return mutator(sol, data)
  
  else:
    print(f'removing a layer from...')
    prtty(sol)
    print('--------------------------------')

    # randomly select a layer to remove (can remove any layer including last)
    hidden_layer_count = sol.configuration['hidden_layer_count']
    loser = rnd.randint(0, hidden_layer_count-1)                    # allow removing any layer including last
    print('old hidden layer count: ', hidden_layer_count)
    print(f'removing layer at index {loser}')
    

    # Copy the original configuration
    layer_names = sol.configuration['layer_names'].copy()
    specifications = sol.configuration['layer_specifications'].copy()
    outputs = sol.configuration['neurons_per_layer'].copy()

    # remove layer information from architecture
    hidden_layer_count -= 1
    layer_names.pop(loser)
    specifications.pop(loser)
    outputs.pop(loser)

    # rebuild hidden layers with new arrangement (same pattern as swap_layers, grow_layer, etc.)
    hidden_layers = []
    input_size = sol.configuration['feature_shape']  # Start with feature_shape
    temporal_removing_layers = ['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D']
    
    for i in range(hidden_layer_count):

      # determine if this is the last layer
      if i == hidden_layer_count - 1:
        # Check if the last layer can remove temporal dimension
        last_layer_type = layer_names[i]
        if last_layer_type not in temporal_removing_layers:
          # Replace with a temporal-removing layer
          replacement_type = rnd.choice(['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D'])
          layer, name, spec, output_size = create_layer(input_size, [replacement_type], specs=None, last_layer=True)
          # Update the layer name and specs
          layer_names[i] = name
          specifications[i] = spec
          if isinstance(output_size, int):
            outputs[i] = output_size
          elif isinstance(output_size, tuple):
            outputs[i] = output_size[1] if len(output_size) > 1 else output_size[0]
          else:
            outputs[i] = 32  # fallback
        else:
          layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
      else:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
      
      hidden_layers.append(layer)
      input_size = output_size  # Update input size for next layer

    # create new Solution with new layer information
    print('new hidden layer count: ', len(hidden_layers))
    new_sol = config_update(
      sol, data, mutator='remove_layer', hidden_layer_count=hidden_layer_count, hidden_layers=hidden_layers, 
      layer_names=layer_names, specifications=specifications, outputs=outputs
    )
  return new_sol


@profile
def swap_layers(sol, data):
    '''
    purpose: agent to swap two layers in the neural network
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with swapped layers
    '''
    print(f'swapping layers in...')
    prtty(sol)
    print('--------------------------------')
    
    # Get current configuration
    hidden_layer_count = sol.configuration['hidden_layer_count']
    layer_names = sol.configuration['layer_names'].copy()
    specifications = sol.configuration['layer_specifications'].copy()
    outputs = sol.configuration['neurons_per_layer'].copy()
    
    # randomly select two different layers to swap (allow swapping with last layer)
    available_indices = list(range(hidden_layer_count))
    
    # need at least 2 layers available to swap
    if len(available_indices) < 2:
        print(f"cannot swap layers - only {len(available_indices)} swappable layer(s) available")
        print('selecting different mutator')
        # randomly select different mutator
        mutator = rnd.choice([
          add_layer, grow_layer, shrink_layer, change_activation, 
          change_optimizer, change_loss_func, change_batch_size,
          change_epochs
        ])
        return mutator(sol, data)
    
    # randomly select two different layers to swap
    idx1, idx2 = rnd.sample(available_indices, 2)
    print(f'swapping layer {idx1} ({layer_names[idx1]}) with layer {idx2} ({layer_names[idx2]})')
    
    # Swap all layer information
    layer_names[idx1], layer_names[idx2] = layer_names[idx2], layer_names[idx1]
    specifications[idx1], specifications[idx2] = specifications[idx2], specifications[idx1]
    outputs[idx1], outputs[idx2] = outputs[idx2], outputs[idx1]

    # rebuild hidden layers with new arrangement
    hidden_layers = []
    input_size = sol.configuration['feature_shape']  # Start with feature_shape
    temporal_removing_layers = ['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D']
    
    for i in range(hidden_layer_count):

      # determine if this is the last layer
      if i == hidden_layer_count - 1:
        # Check if the last layer can remove temporal dimension
        last_layer_type = layer_names[i]
        if last_layer_type not in temporal_removing_layers:
          # Replace with a temporal-removing layer
          replacement_type = rnd.choice(['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D'])
          layer, name, spec, output_size = create_layer(input_size, [replacement_type], specs=None, last_layer=True)
          # Update the layer name and specs
          layer_names[i] = name
          specifications[i] = spec
          if isinstance(output_size, int):
            outputs[i] = output_size
          elif isinstance(output_size, tuple):
            outputs[i] = output_size[1] if len(output_size) > 1 else output_size[0]
          else:
            outputs[i] = 32  # fallback
        else:
          layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
      else:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
      
      hidden_layers.append(layer)
      input_size = output_size  # Update input size for next layer
    
    # Create new solution
    new_sol = config_update(
        sol, data,
        mutator='swap_layers',
        hidden_layer_count=hidden_layer_count,
        hidden_layers=hidden_layers,
        layer_names=layer_names,
        specifications=specifications,
        outputs=outputs
    )
    
    return new_sol


@profile
def grow_layer(sol, data):
    '''
    purpose: agent to increase the number of neurons in a randomly selected layer
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with larger layer
    '''
    print(f'growing a layer in...')
    prtty(sol)
    print('--------------------------------')
    
    # get current configuration
    hidden_layer_count = sol.configuration['hidden_layer_count']
    layer_names = sol.configuration['layer_names'].copy()
    specifications = sol.configuration['layer_specifications'].copy()
    outputs = sol.configuration['neurons_per_layer'].copy()
    
    # randomly select a layer to grow (that has units/filters)
    growable_layers = []
    for i in range(hidden_layer_count):
        layer_name = layer_names[i]
        if layer_name in [
          'Dense', 'LSTM', 'GRU', 'SimpleRNN', 'Conv1D', 'SeparableConv1D', 
          'Conv1DTranspose', 'ConvLSTM1D'
        ]:
            growable_layers.append(i)
    
    if not growable_layers:
        print("no growable layers found")
        print('selecting different mutator')
        # randomly select different mutator
        mutator = rnd.choice([
          add_layer, swap_layers, change_activation, remove_layer,
          change_optimizer, change_loss_func, change_batch_size,
          change_epochs
        ])
        return mutator(sol, data)
            
    
    target_idx = rnd.choice(growable_layers)
    print(f'growing layer {target_idx} ({layer_names[target_idx]})')
    

    layer_name = layer_names[target_idx]
    
    if layer_name in ['Dense', 'LSTM', 'GRU', 'SimpleRNN']:
        old_units = specifications[target_idx]['units']
        new_units = rnd.randint(old_units + 1, old_units * 5)
        specifications[target_idx]['units'] = new_units
        outputs[target_idx] = new_units
        print(f'growing units from {old_units} to {new_units}')

    elif layer_name in ['Conv1D', 'SeparableConv1D', 'Conv1DTranspose', 'ConvLSTM1D']:
        old_filters = specifications[target_idx]['filters']
        new_filters = rnd.randint(old_filters + 1, old_filters * 5)
        specifications[target_idx]['filters'] = new_filters
        outputs[target_idx] = new_filters
        print(f'growing filters from {old_filters} to {new_filters}')
    
    # rebuild all layers from specifications
    hidden_layers = []
    input_size = sol.configuration['feature_shape']
    temporal_removing_layers = ['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D']
    
    for i in range(hidden_layer_count):
        is_last = (i == hidden_layer_count - 1)
        if is_last:
          # Check if the last layer can remove temporal dimension
          last_layer_type = layer_names[i]
          if last_layer_type not in temporal_removing_layers:
            # Replace with a temporal-removing layer
            replacement_type = rnd.choice(['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D'])
            layer, name, spec, out_size = create_layer(input_size, [replacement_type], specs=None, last_layer=True)
            # Update the layer name and specs
            layer_names[i] = name
            specifications[i] = spec
            if isinstance(out_size, int):
              outputs[i] = out_size
            elif isinstance(out_size, tuple):
              outputs[i] = out_size[1] if len(out_size) > 1 else out_size[0]
            else:
              outputs[i] = 32  # fallback
          else:
            layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
        else:
          layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
        hidden_layers.append(layer)
        input_size = out_size
    
    # Create new solution
    new_sol = config_update(
        sol, data, mutator='grow_layer',
        hidden_layer_count=hidden_layer_count,
        hidden_layers=hidden_layers,
        layer_names=layer_names,
        specifications=specifications,
        outputs=outputs
    )
    
    return new_sol


@profile
def shrink_layer(sol, data):
    '''
    purpose: agent to decrease the number of neurons in a randomly selected layer
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with smaller layer
    '''
    print(f'shrinking a layer in...')
    prtty(sol)
    print('--------------------------------')
    
    # get current configuration
    hidden_layer_count = sol.configuration['hidden_layer_count']
    layer_names = sol.configuration['layer_names'].copy()
    specifications = sol.configuration['layer_specifications'].copy()
    outputs = sol.configuration['neurons_per_layer'].copy()
    
    # Randomly select a layer to shrink (that has units/filters)
    shrinkable_layers = []
    for i in range(hidden_layer_count):
        layer_name = layer_names[i]
        if layer_name in ['Dense', 'LSTM', 'GRU', 'SimpleRNN', 'Conv1D', 'SeparableConv1D', 'Conv1DTranspose', 'ConvLSTM1D']:
            shrinkable_layers.append(i)
    
    if not shrinkable_layers:
        print("no shrinkable layers found")
        print('selecting different mutator')
        # randomly select different mutator
        mutator = rnd.choice([
          add_layer, swap_layers, change_activation, remove_layer,
          change_optimizer, change_loss_func, change_batch_size,
          change_epochs
        ])
        return mutator(sol, data)
    
    target_idx = rnd.choice(shrinkable_layers)
    print(f'shrinking layer {target_idx} ({layer_names[target_idx]})')
    

    layer_name = layer_names[target_idx]
    
    if layer_name in ['Dense', 'LSTM', 'GRU', 'SimpleRNN']:
        old_units = specifications[target_idx]['units']
        # Shrink to between max(4, units//5) and units-1 to ensure it's always smaller
        min_units = max(4, old_units // 5)
        max_units = old_units - 1
        if min_units >= max_units:
            # If we can't shrink (e.g., units is 4 or 5), try a different mutator
            print(f"cannot shrink layer {target_idx} - units ({old_units}) too small")
            print('selecting different mutator')
            mutator = rnd.choice([
              add_layer, swap_layers, change_activation, remove_layer,
              change_optimizer, change_loss_func, change_batch_size,
              change_epochs, grow_layer
            ])
            return mutator(sol, data)
        new_units = rnd.randint(min_units, max_units)
        specifications[target_idx]['units'] = new_units
        outputs[target_idx] = new_units
        print(f'shrinking units from {old_units} to {new_units}')

    elif layer_name in ['Conv1D', 'SeparableConv1D', 'Conv1DTranspose', 'ConvLSTM1D']:
        old_filters = specifications[target_idx]['filters']
        # Shrink to between max(4, filters//5) and filters-1 to ensure it's always smaller
        min_filters = max(4, old_filters // 5)
        max_filters = old_filters - 1
        if min_filters >= max_filters:
            # If we can't shrink (e.g., filters is 4 or 5), try a different mutator
            print(f"cannot shrink layer {target_idx} - filters ({old_filters}) too small")
            print('selecting different mutator')
            mutator = rnd.choice([
              add_layer, swap_layers, change_activation, remove_layer,
              change_optimizer, change_loss_func, change_batch_size,
              change_epochs, grow_layer
            ])
            return mutator(sol, data)
        new_filters = rnd.randint(min_filters, max_filters)
        specifications[target_idx]['filters'] = new_filters
        outputs[target_idx] = new_filters
        print(f'shrinking filters from {old_filters} to {new_filters}')
    
    # Rebuild all layers from specifications
    hidden_layers = []
    input_size = sol.configuration['feature_shape']
    temporal_removing_layers = ['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D']
    
    for i in range(hidden_layer_count):
        is_last = (i == hidden_layer_count - 1)
        if is_last:
          # Check if the last layer can remove temporal dimension
          last_layer_type = layer_names[i]
          if last_layer_type not in temporal_removing_layers:
            # Replace with a temporal-removing layer
            replacement_type = rnd.choice(['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D'])
            layer, name, spec, out_size = create_layer(input_size, [replacement_type], specs=None, last_layer=True)
            # Update the layer name and specs
            layer_names[i] = name
            specifications[i] = spec
            if isinstance(out_size, int):
              outputs[i] = out_size
            elif isinstance(out_size, tuple):
              outputs[i] = out_size[1] if len(out_size) > 1 else out_size[0]
            else:
              outputs[i] = 32  # fallback
          else:
            layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
        else:
          layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
        hidden_layers.append(layer)
        input_size = out_size
    
    # create new solution
    new_sol = config_update(
        sol, data, 
        mutator='shrink_layer',
        hidden_layer_count=hidden_layer_count,
        hidden_layers=hidden_layers,
        layer_names=layer_names,
        specifications=specifications,
        outputs=outputs
    )
    
    return new_sol


@profile
def change_activation(sol, data):
    '''
    purpose: agent to change the activation function of a randomly selected layer
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with different activation
    '''
    print(f'changing activation function in...')
    prtty(sol)
    print('--------------------------------')
    
    # Get current configuration
    hidden_layer_count = sol.configuration['hidden_layer_count']
    layer_names = sol.configuration['layer_names'].copy()
    specifications = sol.configuration['layer_specifications'].copy()
    outputs = sol.configuration['neurons_per_layer'].copy()
    
    # Available activations
    activations = [
        'relu', 'tanh', 'sigmoid', 'linear', 'elu', 'selu', 
        'softplus', 'softsign', 'swish', 'gelu'
    ]
    recurrent_activations = ['sigmoid', 'tanh', 'hard_sigmoid']
    
    # Find layers with activation functions
    layers_with_activation = []
    for i in range(hidden_layer_count):
        if specifications[i] is not None and 'activation' in specifications[i]:
            layers_with_activation.append(i)
    
    if not layers_with_activation:
        print("no layers with activation functions found")
        print('selecting different mutator')

        # randomly select different mutator
        mutator = rnd.choice([
          add_layer, swap_layers, remove_layer,
          change_optimizer, change_loss_func, change_batch_size,
          change_epochs, grow_layer, shrink_layer
        ])
        return mutator(sol, data)
    
    target_idx = rnd.choice(layers_with_activation)
    layer_name = layer_names[target_idx]
    print(f'changing activation for layer {target_idx} ({layer_name})')
    
    # change activation
    current_activation = specifications[target_idx]['activation']
    new_activation = rnd.choice([a for a in activations if a != current_activation])
    specifications[target_idx]['activation'] = new_activation
    print(f'changing activation from {current_activation} to {new_activation}')
    
    # Also change recurrent_activation for RNN layers if present
    if 'recurrent_activation' in specifications[target_idx]:
        current_rec_activation = specifications[target_idx]['recurrent_activation']
        new_rec_activation = rnd.choice([a for a in recurrent_activations if a != current_rec_activation])
        specifications[target_idx]['recurrent_activation'] = new_rec_activation
        print(f'changing recurrent activation from {current_rec_activation} to {new_rec_activation}')
    
    # Rebuild all layers from specifications
    hidden_layers = []
    input_size = sol.configuration['feature_shape']
    temporal_removing_layers = ['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D']
    
    for i in range(hidden_layer_count):
        is_last = (i == hidden_layer_count - 1)
        if is_last:
          # Check if the last layer can remove temporal dimension
          last_layer_type = layer_names[i]
          if last_layer_type not in temporal_removing_layers:
            # Replace with a temporal-removing layer
            replacement_type = rnd.choice(['LSTM', 'SimpleRNN', 'GRU', 'GlobalAveragePooling1D', 'GlobalMaxPooling1D'])
            layer, name, spec, out_size = create_layer(input_size, [replacement_type], specs=None, last_layer=True)
            # Update the layer name and specs
            layer_names[i] = name
            specifications[i] = spec
            if isinstance(out_size, int):
              outputs[i] = out_size
            elif isinstance(out_size, tuple):
              outputs[i] = out_size[1] if len(out_size) > 1 else out_size[0]
            else:
              outputs[i] = 32  # fallback
          else:
            layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
        else:
          layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
        hidden_layers.append(layer)
        input_size = out_size
    
    # Create new solution
    new_sol = config_update(
        sol, data, mutator='change_activation',
        hidden_layer_count=hidden_layer_count,
        hidden_layers=hidden_layers,
        layer_names=layer_names,
        specifications=specifications,
        outputs=outputs
    )
    
    return new_sol


@profile
def change_optimizer(sol, data):
    '''
    purpose: agent to change the optimizer
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with different optimizer
    '''
    print(f'changing optimizer in...')
    prtty(sol)
    print('--------------------------------')
    
    # Available optimizers
    optimizers = ['adamax', 'adadelta','ftrl', 'lamb',  'nadam', 'rmsprop', 
                'sgd', 'adagrad', 'lion',  'adamw', 'adafactor', 'adam']                                
    
    
    # Select a different optimizer
    current_optimizer = sol.configuration['optimizer']
    new_optimizer = rnd.choice([opt for opt in optimizers if opt != current_optimizer])
    print(f'changing optimizer from {current_optimizer} to {new_optimizer}')
    
    # Create new solution with updated optimizer
    new_sol = config_update(sol, data, mutator='change_optimizer', optimizer=new_optimizer)
    
    return new_sol


@profile
def change_epochs(sol, data):
    '''
    purpose: agent to change the number of training epochs
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with different epoch count
    '''
    print(f'changing epochs in...')
    prtty(sol)
    print('--------------------------------')
    
    current_epochs = sol.configuration['epochs']
    
    # randomly increase or decrease epochs
    change = rnd.randint(-20, 20)
    new_epochs = max(3, current_epochs + change)
    
    print(f'changing epochs from {current_epochs} to {new_epochs}')
    
    # Create new solution with updated epochs
    new_sol = config_update(sol, data, mutator='change_epochs', epochs=new_epochs)
    
    return new_sol


@profile
def change_batch_size(sol, data):
    '''
    purpose: agent to change the batch size
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with different batch size
    '''
    print(f'changing batch size in...')
    prtty(sol)
    print('--------------------------------')
    
    current_batch_size = sol.configuration['batch_size']
    
    # Randomly increase or decrease batch size by 10-50
    change = rnd.randint(-50, 50)
    new_batch_size = max(32, current_batch_size + change)
    
    print(f'changing batch size from {current_batch_size} to {new_batch_size}')
    
    # create new solution with updated batch size
    new_sol = config_update(sol, data, mutator='change_batch_size', batch_size=new_batch_size)
    
    return new_sol


@profile
def change_loss_func(sol, data):
    '''
    purpose: agent to change the loss function
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with different loss function
    '''
    print(f'changing loss function in...')
    prtty(sol)
    print('--------------------------------')
    
    # available loss functions
    loss_functions = ['categorical_crossentropy', 'categorical_focal_crossentropy', 'kl_divergence']
    
    # select a different loss function
    current_loss = sol.configuration['loss_function']
    new_loss = rnd.choice([loss for loss in loss_functions if loss != current_loss])
    print(f'changing loss function from {current_loss} to {new_loss}')
    
    # Create new solution with updated loss function
    new_sol = config_update(sol, data, mutator='change_loss_func', loss_func=new_loss)
    
    return new_sol

  
@profile
def crossover(sol1, sol2, data):
    '''
    purpose: agent to crossover two solutions
    params:
        sol1: Solution object containing configuration, model, and metrics
        sol2: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits
    returns: new Solution Object with crossover of two solutions
    '''
    print(f'crossover in...')
    prtty(sol1)
    prtty(sol2)
    print('--------------------------------')

    # extract architecture from solution 1
    hidden_layer_count = sol1.configuration['hidden_layer_count']
    hidden_layers = sol1.configuration['hidden_layers']
    layer_names = sol1.configuration['layer_names']
    specifications = sol1.configuration['layer_specifications']
    outputs = sol1.configuration['neurons_per_layer']

    # extract hyperparameters from solution 2
    loss_func = sol2.configuration['loss_function']
    optimizer = sol2.configuration['optimizer']
    epochs = sol2.configuration['epochs']
    batch_size = sol2.configuration['batch_size']

    # genetic information
    id = uuid.uuid4()
    id1 = sol1.configuration['id']
    id2 = sol2.configuration['id']
    parent_id = f'{id1} & {id2}'

    # create new solution with crossover of architecture and hyperparameters
    new_sol = config_update(
      sol1, data, mutator='crossover', 
      hidden_layer_count=hidden_layer_count, hidden_layers=hidden_layers, 
      layer_names=layer_names, specifications=specifications, outputs=outputs, 
      loss_func=loss_func, optimizer=optimizer, epochs=epochs, batch_size=batch_size, 
      id=id, parent_id=parent_id
    )

    return new_sol


