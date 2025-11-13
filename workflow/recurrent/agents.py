from solution import Solution
import random as rnd
from profiler import profile
import random as rnd
from tensorflow.keras import layers


def create_layer(input_size, layer_types=[], last_layer=False, specs=None):

  # intialize variables (removed any activation that required too much stipulations)
  activations = [
    'relu', 'tanh', 'sigmoid', 'linear', 'celu', 'elu', 'hard_silu', 'soft_shrink',
    'hard_shrink', 'softmax', 'tanh_shrink'
    # 'celu', 'elu',
    # 'hard_sigmoid', 'hard_silu', 'hard_tanh', 'leaky_relu',
    # 'mish', 'selu', 'silu', 'softmax', 'tanh', 'tanh_shrink'
  ]

  specifications = None
           

  # select layer type
  if len(layer_types) > 0:
    layer = rnd.choice(layer_types)

  else:
    layer = rnd.choice([
        'Dropout',
        'Normalization', 'SpectralNormalization',
        'SeparableConv1D', 'Conv1D', 'Conv1DTranspose',                                                  # Convolution Layers         
        'MaxPooling1D', 'AveragePooling1D',                                                              # Pooling Layers
        'LSTM', 'SimpleRNN', 'GRU', 'Bidirectional', 'ConvLSTM1D'                                        # Recurrent Layers
    ])

  # ensure last temporal layer removes time steps
  if last_layer == True:
    return_sequences = False
  else:
    return_sequences = True   

  if last_layer and layer in ['LSTM', 'SimpleRNN', 'GRU']:
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
    specifications = f'pool_size : {pool_size} - strides : {strides} - padding : {padding}'
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
      recurrent_activation = rnd.choice(activations)   
    else:
      recurrent_dropout = specs['recurrent_dropout']
      activation = specs['activation']
      recurrent_activation = specs['recurrent_activation']

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
      recurrent_activation = rnd.choice(activations)
    else:
      recurrent_dropout = specs['recurrent_dropout']
      activation = specs['activation']
      recurrent_activation = specs['recurrent_activation']

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
    specifications['rnn_layer'] = rnn_layer
    specifications['return_sequences'] = return_sequences
    for k in specs.keys():
      specifications[f'{k}'] = k
       
    tf_layer = layers.Bidirectional(rnn_layer, merge_mode = merge_mode)


  # Performs spectral normalization on the weights of a target layer
  elif layer == 'SpectralNormalization':
    layer_with_kernel, name, specs, output_size = create_layer(input_size, ['Conv1D', 'GRU'])
    if layer_with_kernel is None:
        raise ValueError("SpectralNormalization wrapper requires a valid layer_with_kernel")

    specifications = {}
    specifications['target_layer'] = name
    specifications['return_sequences'] = return_sequences
    for k in specs.keys():
      specifications[f'{k}'] = k

    tf_layer = layers.SpectralNormalization(layer_with_kernel)


  else:
    raise ValueError(f"Layer '{layer}' not implemented in create_layer.")

  return tf_layer, layer, specifications, output_size


def transfer_weights(parent, child, data):
    """
    build child model architecture and transfer compatible weights from parent
    BEFORE developing model from scratch to speed up dev
    """
    try:
      print('transfering parent model weights before training')
      # build the child model architecture
      child._build_model_architecture()
      
      # Get layers from both models (excluding input/output layers)
      parent_layers = parent.model.layers
      child_layers = child.model.layers

      parent_layer_names = parent.configuration['layer_names']
      child_layer_names = child.configuration['layer_names']

      parent_layer_specs = parent.configuration['layer_specifications']
      child_layer_specs = child.configuration['layer_specifications']

      print(f"Parent layer: {len(parent_layers)} - {parent_layer_names} - {parent_layer_specs}")
      print(f"Child layers: {len(child_layers)} - {child_layer_names} - {child_layer_specs}")

      
      # transfer weights for compatible layers
      min_layers = min(len(parent_layer_names), len(child_layer_names))
      transferred_count = 0
      for i in range(min_layers):

        # Skip if layer names don't match
        if parent_layer_names[i] != child_layer_names[i]:
            print(f"⚠️ Layer name mismatch at {i}: {parent_layer_names[i]} vs {child_layer_names[i]}")
            continue
        
        # Skip if layer specifications don't match (when both are not None)
        if (parent_layer_specs[i] is not None and child_layer_specs[i] is not None and 
            parent_layer_specs[i] != child_layer_specs[i]):
            print(f"⚠️ Layer spec mismatch at {i}: {parent_layer_specs[i]} vs {child_layer_specs[i]}")
            continue
            

        
        try:
            # Model layer index is offset by 1 (InputLayer is at index 0)
            model_layer_idx = i + 1

            # Skip input layer (usually index 0) and output layer
            if model_layer_idx >= len(parent_layers) or model_layer_idx >= len(child_layers):
                print(f"⚠️ Layer index {i} out of range for model layers")
                continue
            
            # Get weights from parent layer
            parent_weights = parent_layers[model_layer_idx].get_weights()

            if not parent_weights:  # Skip layers without weights (e.g., Dropout, Pooling)
                print(f"⚠️ No weights to transfer for layer {i}: {parent_layer_names[i]}")
                continue
            
            child_weights = child_layers[model_layer_idx].get_weights()

            # Check if child layer is built and has compatible shape
            # if not child_layers[i].built:
            #     print(f"⚠️ Child layer {i} not built, building it...")
            #     try:
            #       import tensorflow as tf
            #       # Build the layer by running a forward pass
            #       if hasattr(child_layers[i], 'input_shape') and child_layers[i].input_shape:
            #           test_input = tf.ones((1,) + child_layers[i].input_shape[1:])
            #           _ = child_layers[i](test_input)
            #     except Exception as build_error:
            #       print(f"❌ Could not build child layer {i}: {build_error}")
            #       continue
          
            # Check if weight shapes match
            if len(parent_weights) == len(child_weights):
                shapes_match = all(pw.shape == cw.shape for pw, cw in zip(parent_weights, child_weights))
                       
                # for pw, cw in zip(parent_weights, child_weights):
                #   if pw.shape != cw.shape:
                #       shapes_match = False
                #       print(f"⚠️ Shape mismatch at layer {i}: {pw.shape} vs {cw.shape}")
                #       break
                 
                if shapes_match:
                    child_layers[i].set_weights(parent_weights)
                    transferred_count += 1
                    print(f"✅ Transferred weights for layer {i}: {parent_layer_names[i]}")
                else:
                    print(f"❌ Shape mismatch, skipping layer {i}")
            else:
                  print(f"❌ Weight count mismatch at layer {i}")
                  
        except Exception as e:
            print(f"❌ Could not transfer weights for layer {i}: {e}")

  
      print(f"Successfully transferred {transferred_count}/{min_layers} layers")
        
      # If we transferred any weights, fine-tune. Otherwise train from scratch.
      if transferred_count > 0:
          print(f"Fine-tuning with {transferred_count} transferred layers")
          child._compile_and_finetune(data)
      else:
          print("No weights transferred, training from scratch")
          child.develop_model(data)
        
      print(f"Successfully transferred {transferred_count}/{len(parent_layers)} layers")
        

    except Exception as e:
        print(f"Weight transfer failed: {e}. Training from scratch.")
        import traceback
        traceback.print_exc()  # This will show the full error traceback
        child.develop_model(data)


def config_update(
    sol, data, hidden_layer_count=None, hidden_layers=None, layer_names=None, specifications=None, outputs=None, 
    loss_func=None, optimizer=None, epochs=None, batch_size=None, weight_transfer=True
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
    'feature_shape': sol.configuration['feature_shape'],
    'class_count': sol.configuration['class_count']
  }


  # generate new solution
  new_sol = Solution(new_configs)

  # transfer weights BEFORE training
  if sol.model is not None and weight_transfer:
      transfer_weights(sol, new_sol, data)
  else:
      # if no parent model, train from scratch
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
  print(f'adding a layer to {sol}')
  # get input size of second to last hidden layer to create a new second to last hidden layer (will add it after)
  neurons_per_layer = sol.configuration['neurons_per_layer']
  prev_layer_size = neurons_per_layer[-2]

  # create new layer
  new_layer, name, specs, output_size = create_layer(prev_layer_size, last_layer=False)

  # add new layer information to architecture configuration
  hidden_layer_count = sol.configuration['hidden_layer_count'] + 1
  hidden_layers = sol.configuration['hidden_layers'].copy()
  layer_names = sol.configuration['layer_names'].copy()
  specifications = sol.configuration['layer_specifications'].copy()
  outputs = sol.configuration['neurons_per_layer'].copy()
  
  hidden_layers.insert(-1, new_layer)
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
     return sol
  
  else:
    print(f'removing a layer from {sol}')

    # randomly select a layer to remove
    hidden_layer_count = sol.configuration['hidden_layer_count']
    loser = rnd.randint(0, hidden_layer_count-1)
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

    hidden_layers = []
    input_size = sol.configuration['feature_shape']
    for i in range(hidden_layer_count):

      # determine if this is the last layer
      if i == hidden_layer_count - 1:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
      else:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
      
      hidden_layers.append(layer)
      input_size = output_size

    # create new Solution with new layer information
    print('new hidden layer count: ', len(hidden_layers))
    new_sol = config_update(
      sol, data, hidden_layer_count=hidden_layer_count, hidden_layers=hidden_layers, 
      layer_names=layer_names, specifications=specifications, outputs=outputs, weight_transfer=False
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
    if sol.configuration['hidden_layer_count'] <= 1:
        print("Cannot swap layers - need at least 2 hidden layers")
        return sol
    
    print(f'swapping layers in {sol}')
    
    # Get current configuration
    hidden_layer_count = sol.configuration['hidden_layer_count']
    layer_names = sol.configuration['layer_names'].copy()
    specifications = sol.configuration['layer_specifications'].copy()
    outputs = sol.configuration['neurons_per_layer'].copy()
    
    # randomly select two different layers to swap
    available_indices = list(range(hidden_layer_count))
    idx1, idx2 = rnd.sample(available_indices, 2)
    
    print(f'Swapping layer {idx1} ({layer_names[idx1]}) with layer {idx2} ({layer_names[idx2]})')
    
    # Swap all layer information
    layer_names[idx1], layer_names[idx2] = layer_names[idx2], layer_names[idx1]
    specifications[idx1], specifications[idx2] = specifications[idx2], specifications[idx1]
    outputs[idx1], outputs[idx2] = outputs[idx2], outputs[idx1]

    # rebuild hidden layers with new arrangement
    hidden_layers = []
    input_size = sol.configuration['feature_shape']
    for i in range(hidden_layer_count):

      # determine if this is the last layer
      if i == hidden_layer_count - 1:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i], last_layer=True)
      else:
        layer, name, spec, output_size = create_layer(input_size, [layer_names[i]], specs=specifications[i])
      
      hidden_layers.append(layer)
    
    # Create new solution
    new_sol = config_update(
        sol, data,
        hidden_layer_count=hidden_layer_count,
        hidden_layers=hidden_layers,
        layer_names=layer_names,
        specifications=specifications,
        outputs=outputs, weight_transfer=False
    )
    
    return new_sol


