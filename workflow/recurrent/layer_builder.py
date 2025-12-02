"""
Helper module to rebuild Keras layers from configuration.
This is needed because hidden_layers (Keras layer objects) cannot be serialized to JSON.
"""
from tensorflow.keras import layers
from mutators import create_layer


def rebuild_layers_from_config(config):
    """
    Rebuild hidden_layers list from layer_names and layer_specifications.
    
    Args:
        config: Configuration dictionary with:
            - layer_names: List of layer type names (e.g., ['LSTM', 'Dense'])
            - layer_specifications: List of specification dicts for each layer
            - feature_shape: Input shape tuple
            - neurons_per_layer: List of output sizes for each layer
    
    Returns:
        List of Keras layer objects
    """
    layer_names = config.get('layer_names', [])
    layer_specs = config.get('layer_specifications', [])
    feature_shape = config.get('feature_shape')
    neurons_per_layer = config.get('neurons_per_layer', [])
    
    if not layer_names or not layer_specs:
        raise ValueError("layer_names and layer_specifications must be present in config")
    
    hidden_layers = []
    input_size = feature_shape
    
    for i, (layer_name, specs) in enumerate(zip(layer_names, layer_specs)):
        is_last = (i == len(layer_names) - 1)
        
        # Determine layer types to choose from based on layer_name
        layer_types = [layer_name]
        
        # Rebuild the layer using create_layer with the exact specs
        layer_obj, _, _, output_size = create_layer(
            input_size=input_size,
            layer_types=layer_types,
            last_layer=is_last,
            specs=specs
        )
        
        hidden_layers.append(layer_obj)
        
        # Update input_size for next layer
        if neurons_per_layer and i < len(neurons_per_layer):
            input_size = neurons_per_layer[i]
        else:
            input_size = output_size
    
    return hidden_layers

