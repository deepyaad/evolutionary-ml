from solution import Solution
import random as rnd
from profiler import profile
import random as rnd
from tensorflow.keras import layers
from pprint import pprint as prtty


def create_layer(input_size, layer_types=[], last_layer=False, specs=None):

  # intialize variables
  activations = [
    'celu', 'elu', 'gelu', 'hard_sigmoid', 'hard_shrink', 'hard_tanh', 'hard_silu', 
    'leaky_relu', 'linear', 'mish', 'relu', 'selu', 'silu', 
    'sigmoid', 'softmax', 'softplus', 'softsign', 'soft_shrink', 'swish', 'tanh',
    'tanh_shrink'
  ]

  recurrent_activations = [
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
    layer = rnd.choice([
        'Dropout', 'Dense',                                                                              # Dense Layers          TODO: allow model to add Dense & Dropout Layers after Global Pooling and last Recurrent layers
        'Normalization', 'SpectralNormalization',                                                        # Normalization Layers  TODO: add layers.LayerNormalization()
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

  
  # Dot-product attention layer, a.k.a. Luong-style attention
  elif layer == 'Attention':
    if specs is None:
      dropout = rnd.random() / 3
      score_mode = rnd.choice(['dot', 'concat'])
    else:
      dropout = specs['dropout']
      score_mode = specs['score_mode']

    specifications = {}
    specifications['dropout'] = dropout
    specifications['score_mode'] = score_mode
    tf_layer = layers.Attention()
    output_size = input_size
      


  # preprocessing layer that normalizes continuous features
  elif layer == 'Normalization':
    tf_layer = layers.Normalization()
    output_size = input_size
  

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

  # 1D convolution LSTM (input and recurrent transformations are convolutional)
  elif layer == 'ConvLSTM1D':
    filters = rnd.randint(8, 32)
    if specs is None: 
      kernel_size = rnd.randint(3, 24)
      activation = rnd.choice(activations)
      recurrent_activation = rnd.choice(recurrent_activations)
    else:
      kernel_size = specs.get('kernel_size', rnd.randint(3, 24))
      activation = specs.get('activation', rnd.choice(activations))
      recurrent_activation = specs.get('recurrent_activation', rnd.choice(recurrent_activations))

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
  

  # Long Short-Term Memory layer - Hochreiter 1997
  elif layer == 'LSTM':
    units = rnn_units
    if specs is None: 
      recurrent_dropout = 0
      activation = rnd.choice(activations)
      recurrent_activation = rnd.choice(recurrent_activations)   
    else:
      recurrent_dropout = specs.get('recurrent_dropout', 0)
      activation = specs.get('activation', rnd.choice(activations))
      recurrent_activation = specs.get('recurrent_activation', rnd.choice(recurrent_activations))

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
  

  # Gated Recurrent Unit - Cho et al. 2014
  elif layer == 'GRU':
    units = rnn_units
    if specs is None: 
      recurrent_dropout = 0
      activation = rnd.choice(activations)
      recurrent_activation = rnd.choice(recurrent_activations)
    else:
      recurrent_dropout = specs.get('recurrent_dropout', 0)
      activation = specs.get('activation', rnd.choice(activations))
      recurrent_activation = specs.get('recurrent_activation', rnd.choice(recurrent_activations))

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


  else:
    raise ValueError(f"Layer '{layer}' not implemented in create_layer.")

  return tf_layer, layer, specifications, output_size


def config_update(
    sol, data, hidden_layer_count=None, hidden_layers=None, layer_names=None, specifications=None, outputs=None, 
    loss_func=None, optimizer=None, epochs=None, batch_size=None, feature_selection=None
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

    # feature selection
    'feature_selection': feature_selection if feature_selection else sol.configuration['feature_selection'],

    # input data
    'input_size': sol.configuration['input_size'],
    'output_size': sol.configuration['output_size'],
    'feature_shape': sol.configuration['feature_shape'],
    'class_count': sol.configuration['class_count'],
    'labels_inorder': sol.configuration['labels_inorder'],
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

  # get input size of second to last hidden layer to create a new second to last hidden layer (will add it after)
  neurons_per_layer = sol.configuration['neurons_per_layer']
  
  # handle case where there's only 1 layer (can't use [-2])
  if len(neurons_per_layer) == 1:
    # when there's only 1 layer, the new layer should use feature_shape as input
    # feature_shape is a tuple like (469, 96), use it directly for create_layer
    prev_layer_size = sol.configuration['feature_shape']
  else:
    prev_layer_size = neurons_per_layer[-2]

  # create new layer
  new_layer, name, specs, output_size = create_layer(prev_layer_size, last_layer=False)

  # add new layer information to architecture configuration
  hidden_layer_count = sol.configuration['hidden_layer_count'] + 1
  layer_names = sol.configuration['layer_names'].copy()
  specifications = sol.configuration['layer_specifications'].copy()
  outputs = sol.configuration['neurons_per_layer'].copy()
  
  # Insert new layer specs before last layer
  layer_names.insert(-1, name)
  specifications.insert(-1, specs)

  # Extract integer from output_size (handle both tuple and int)
  if isinstance(output_size, tuple):
      output_int = output_size[1]  # Get feature dimension from tuple
  elif isinstance(output_size, int):
      output_int = output_size  # Already an integer
  else:
      raise ValueError(f"Unexpected output_size type: {type(output_size)}")
  outputs.insert(-1, output_int)

  # Rebuild all layers from specifications (don't reuse old layer objects!)
  hidden_layers = []
  input_size = sol.configuration['feature_shape'][1]
  for i in range(hidden_layer_count):
    is_last = (i == hidden_layer_count - 1)
    layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=is_last)
    hidden_layers.append(layer)
    input_size = out_size

  # create new Solution with new layer information
  new_sol = config_update(
    sol, data, hidden_layer_count=hidden_layer_count, hidden_layers=hidden_layers, 
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

    # randomly select a layer to remove
    hidden_layer_count = sol.configuration['hidden_layer_count']
    loser = rnd.randint(0, hidden_layer_count-2)                    # exclude last hidden layer to keep temporal dimension
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
    input_size = outputs[0] if outputs else sol.configuration.get('input_size', [32])[0]  # Start with first layer's output size
    for i in range(hidden_layer_count):

      # determine if this is the last layer
      if i == hidden_layer_count - 1:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
      else:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
      
      hidden_layers.append(layer)
      input_size = output_size  # Update input size for next layer

    # create new Solution with new layer information
    print('new hidden layer count: ', len(hidden_layers))
    new_sol = config_update(
      sol, data, hidden_layer_count=hidden_layer_count, hidden_layers=hidden_layers, 
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
    
    # randomly select two different layers to swap (exclude last hidden layer)
    available_indices = list(range(hidden_layer_count - 1))
    
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
    input_size = outputs[0] if outputs else sol.configuration['input_size'][0] # Start with first layer's output size
    for i in range(hidden_layer_count):

      # determine if this is the last layer
      if i == hidden_layer_count - 1:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
      else:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
      
      hidden_layers.append(layer)
      input_size = output_size  # Update input size for next layer
    
    # Create new solution
    new_sol = config_update(
        sol, data,
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
    
    # Increase the units/filters by 10-200%
    growth_factor = 1 + rnd.uniform(0.1, 2)
    layer_name = layer_names[target_idx]
    
    if layer_name in ['Dense', 'LSTM', 'GRU', 'SimpleRNN']:
        new_units = int(specifications[target_idx]['units'] * growth_factor)
        specifications[target_idx]['units'] = new_units
        outputs[target_idx] = new_units
        print(f'growing units from {specifications[target_idx]["units"]} to {new_units}')

    elif layer_name in ['Conv1D', 'SeparableConv1D', 'Conv1DTranspose', 'ConvLSTM1D']:
        new_filters = int(specifications[target_idx]['filters'] * growth_factor)
        specifications[target_idx]['filters'] = new_filters
        outputs[target_idx] = new_filters
        print(f'growing filters from {specifications[target_idx]["filters"]} to {new_filters}')
    
    # rebuild all layers from specifications
    hidden_layers = []
    input_size = outputs[0]
    for i in range(hidden_layer_count):
        is_last = (i == hidden_layer_count - 1)
        layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=is_last)
        hidden_layers.append(layer)
        input_size = out_size
    
    # Create new solution
    new_sol = config_update(
        sol, data,
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
    
    # Decrease the units/filters by 10-40% (but keep at least 4)
    shrink_factor = 1 - rnd.uniform(0.1, 0.4)
    layer_name = layer_names[target_idx]
    
    if layer_name in ['Dense', 'LSTM', 'GRU', 'SimpleRNN']:
        new_units = max(4, int(specifications[target_idx]['units'] * shrink_factor))
        specifications[target_idx]['units'] = new_units
        outputs[target_idx] = new_units
        print(f'shrinking units from {specifications[target_idx]["units"]} to {new_units}')

    elif layer_name in ['Conv1D', 'SeparableConv1D', 'Conv1DTranspose', 'ConvLSTM1D']:
        new_filters = max(4, int(specifications[target_idx]['filters'] * shrink_factor))
        specifications[target_idx]['filters'] = new_filters
        outputs[target_idx] = new_filters
        print(f'shrinking filters from {specifications[target_idx]["filters"]} to {new_filters}')
    
    # Rebuild all layers from specifications
    hidden_layers = []
    input_size = outputs[0] if outputs else sol.configuration.get('input_size', [32])[0]
    for i in range(hidden_layer_count):
        is_last = (i == hidden_layer_count - 1)
        layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=is_last)
        hidden_layers.append(layer)
        input_size = out_size
    
    # create new solution
    new_sol = config_update(
        sol, data,
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
    input_size = outputs[0]
    for i in range(hidden_layer_count):
        is_last = (i == hidden_layer_count - 1)
        layer, _, _, out_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=is_last)
        hidden_layers.append(layer)
        input_size = out_size
    
    # Create new solution
    new_sol = config_update(
        sol, data,
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
    new_sol = config_update(sol, data, optimizer=new_optimizer)
    
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
    change = rnd.randint(-10, 10)
    new_epochs = max(3, current_epochs + change)
    
    print(f'changing epochs from {current_epochs} to {new_epochs}')
    
    # Create new solution with updated epochs
    new_sol = config_update(sol, data, epochs=new_epochs)
    
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
    new_sol = config_update(sol, data, batch_size=new_batch_size)
    
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
    new_sol = config_update(sol, data, loss_func=new_loss)
    
    return new_sol


@profile
def change_feature_selection(sol, data):
    '''
    purpose: agent to change the feature selection
    params:
        sol: Solution object containing configuration, model, and metrics
        data: data dictionary with training and testing splits or LazyDataLoader
    returns: new Solution Object with different feature selection
    '''
    from data_loader import LazyDataLoader
    
    print(f'changing feature selection in...')
    prtty(sol)
    print('--------------------------------')
    
    # available feature selections - check which are actually available
    if isinstance(data, LazyDataLoader):
        # Only use features that are actually available
        feature_selections = [fs for fs in LazyDataLoader.AVAILABLE_FEATURES 
                             if data.is_feature_available(fs)]
        if not feature_selections:
            raise ValueError("No available features found in data loader")
    else:
        # Legacy format - use all possible features
        feature_selections = [
          'stft', 'mel_specs', 'mfccs', 'mctct', 'parakeet', 
          'seamlessM4T', 'whisper'
        ]
    
    # select a different feature selection
    current_feature_selection = sol.configuration['feature_selection']
    available_selections = [fs for fs in feature_selections if fs != current_feature_selection]
    
    if not available_selections:
        print(f"no alternative feature selections available, keeping {current_feature_selection}")
        # Try a different mutator instead
        mutator = rnd.choice([
          add_layer, swap_layers, remove_layer, grow_layer, shrink_layer,
          change_activation, change_optimizer, change_loss_func, 
          change_batch_size, change_epochs
        ])
        return mutator(sol, data)
    
    new_feature_selection = rnd.choice(available_selections)
    print(f'changing feature selection from {current_feature_selection} to {new_feature_selection}')
    
    # Get new feature shape if using LazyDataLoader
    if isinstance(data, LazyDataLoader):
        new_feature_shape = data.get_feature_shape(new_feature_selection)
        old_feature_shape = sol.configuration.get('feature_shape')
        
        # If feature shape changed, we need to rebuild layers
        if old_feature_shape != new_feature_shape:
            print(f'Feature shape changed: {old_feature_shape} -> {new_feature_shape}')
            # Rebuild layers for new feature shape
            hidden_layer_count = sol.configuration['hidden_layer_count']
            hidden_layers, layer_names, specifications, outputs = sol._rebuild_layers_for_feature_shape(
                new_feature_shape, hidden_layer_count
            )
            
            # Create new solution with updated feature selection and rebuilt layers
            new_sol = config_update(
                sol, data, 
                feature_selection=new_feature_selection,
                hidden_layers=hidden_layers,
                layer_names=layer_names,
                specifications=specifications,
                outputs=outputs
            )
            # Update feature_shape in the new solution
            new_sol.configuration['feature_shape'] = new_feature_shape
        else:
            # Feature shape is the same, just change feature selection
            new_sol = config_update(sol, data, feature_selection=new_feature_selection)
    else:
        # Legacy format - assume feature_shape doesn't change
        new_sol = config_update(sol, data, feature_selection=new_feature_selection)
    
    return new_sol


