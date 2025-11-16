"""
@author: Ananda Francis
@file: connections.py: familiarize myself with different wants to use Keras Functional API to expand architecture search space
"""

from tensorflow.keras.layers import (
    Add, Subtract, Multiply, Average, Maximum, Minimum, 
    Concatenate, Dot, Dot, Flatten, Reshape,
    BatchNormalization, Dropout, Activation,
    Dense, Lambda, Input, Conv1D, MaxPooling1D, UpSampling1D, 
    Concatenate, Dense, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
import tensorflow as tf

def main():

    # sequential / chain connection: standard layer stacking, fully connected layers (one goes into the next)
    inputs = Input(shape=(100,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    seq_model = Model(inputs=inputs, outputs=x)
    


    # skip connections / residual: outputs from earlier layers into later layers (common in residual networks)
    inputs = Input(shape=(256,))
    x1 = Dense(128, activation='relu')(inputs)
    x2 = Dense(128, activation='linear')(x1)                # often linear before adding
    if x1.shape[-1] == x2.shape[-1]:                        # check for same dimensions
        skip = Add()([x1, x2])                              # ResidualBlock()
    else:                                                   # projection if different dimensions
        projected_inputs = Dense(128)(inputs)
        skip = Add()([projected_inputs, x2])                # ResidualBlock()
    skip_model = Model(inputs=inputs, outputs=skip)



    # dense skip connection: each layer receives concatenated features from all previous layers
    inputs = Input(shape=(256,))
    x1 = Dense(64, activation='relu')(inputs)
    x2_input = Concatenate()([inputs, x1])
    x2 = Dense(64, activation='relu')(x2_input)
    x3_input = Concatenate()([inputs, x1, x2])
    x3 = Dense(64, activation='relu')(x3_input)
    final_input = Concatenate()([inputs, x1, x2, x3])           # Final layer uses all feature maps
    dense_skip_model = Model(inputs=inputs, outputs=final_input)



    # multiplicative / gated connections: outputs are combined via element-wise multiplication
    inputs = Input(shape=(100,))
    a = Dense(64, activation='relu')(inputs)
    b = Dense(64, activation='sigmoid')(inputs)
    gated = Multiply()([a, b])                                  # Lambda()
    gated_model = Model(inputs=inputs, outputs=gated)



    # simple branching layers into multiple paths than merging
    inputs = Input(shape=(100,))
    branch_a = Dense(32, activation='relu')(inputs)
    branch_b = Dense(32, activation='tanh')(inputs)
    merged = Concatenate()([branch_a, branch_b])                # Add(), Multiply(), etc
    branch_model = Model(inputs=inputs, outputs=merged)



    # complex branching & merging (parallel pathways)
    inputs = Input(shape=(100,))    
    branch_a = Dense(64, activation='relu')(inputs)                                 # Branch A: Deeper network
    branch_a = Dense(32, activation='relu')(branch_a)
    branch_a = Dense(16, activation='relu')(branch_a)
    branch_a_proj = Dense(32, activation='relu')(branch_a)                          # Project to 32  
    branch_b = Dense(96, activation='tanh')(inputs)                                 # Branch B: Wider but shallower network
    branch_b = Dense(48, activation='tanh')(branch_b)
    branch_b_proj = Dense(32, activation='tanh')(branch_b)                          # Project to 32
    branch_c = Dense(32, activation='sigmoid')(inputs)                              # Branch C: Different activation
    branch_c = Dense(32, activation='sigmoid')(branch_c)
    merged_concat = Concatenate()([branch_a, branch_b, branch_c])                   # Most common for preserving all features
    merged_add = Add()([branch_a_proj, branch_b_proj, branch_c])                    # Requires matching shapes, fuses information
    merged_avg = Average()([branch_a_proj, branch_b_proj, branch_c])                # Requires matching shapes
    merged_mul = Multiply()([branch_a_proj, branch_b_proj, branch_c])               # Requires matching shapes
    complex_branch_model = Model(inputs=inputs, outputs=merged_concat)



    # siamese network sharing 2 layers
    input_a = Input(shape=(100,))                                                       # Two identical branches sharing weights
    input_b = Input(shape=(100,))
    shared_encoder_1 = Dense(128, activation='relu')                                    # Shared tower - same weights for both inputs
    shared_encoder_2 = Dense(64, activation='relu')
    encoded_a = shared_encoder_1(input_a)                                               # Process both inputs through shared layers
    encoded_a = shared_encoder_2(encoded_a)
    encoded_b = shared_encoder_1(input_b)
    encoded_b = shared_encoder_2(encoded_b)
    l1_distance = Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_a, encoded_b])         # Distance metric between processed inputs
    siamese_model = Model(inputs=[input_a, input_b], outputs=l1_distance)



    # siamese networks sharing 1 layer
    input_a = Input(shape=(784,))
    input_b = Input(shape=(784,))
    shared_encoder = Dense(128, activation='relu')                                                # Define the base network/encoder once
    encoded_a = shared_encoder(input_a)                                                           # Process both inputs with the same shared weights
    encoded_b = shared_encoder(input_b)
    distance_l1 = Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_a, encoded_b])                   # Comparison/Distance Layer (using a L1 Distance (Absolute Difference)
    distance_l2 = Lambda(lambda x: tf.square(x[0] - x[1]))([encoded_a, encoded_b])                # L2 Distance (Euclidean)
    distance_dot = Dot(axes=1, normalize=True)([encoded_a, encoded_b])                            # Cosine Similarity
    siamese_model = Model(inputs=[input_a, input_b], outputs=distance_l1)                         # Final classifier on the distance vector



    # attention mechanism
    inputs = Input(shape=(None, 256))                                                                       # Sequence input
    query = Dense(64)(inputs)                                                                               # Self-attention: each position attends to all positions
    key = Dense(64)(inputs)
    value = Dense(64)(inputs)                                                                               # TODO: Check that if this should be 64 or 256
    attention_scores = Dot(axes=(2, 2))([query, key])                                                       # Q*K^T
    attention_scores = Lambda(lambda x: x / tf.math.sqrt(tf.cast(64, tf.float32)))(attention_scores)        # Scaled dot-product attention
    attention_weights = Activation('softmax')(attention_scores)
    attended = Dot(axes=(2, 1))([attention_weights, value])                                                 # Apply attention to values
    if attended.shape[-1] == inputs.shape[-1]:                                                              # Residual connection around attention
        attended = Add()([inputs, attended])
    attention_model = Model(inputs=inputs, outputs=attended)



    # u-net style (encoder-decoder with skip connections) (ensure this is spatial-temporal data not 1D tabular)
    inputs = Input(shape=(256,))
    enc1 = Dense(128, activation='relu')(inputs)                            # Encoder
    enc2 = Dense(64, activation='relu')(enc1)
    bottleneck = Dense(32, activation='relu')(enc2)
    dec1 = Dense(64, activation='relu')(bottleneck)                         # Decoder with skip connections
    dec1_with_skip = Concatenate()([dec1, enc2])                            # Skip connection from encoder to decoder
    dec2 = Dense(128, activation='relu')(dec1_with_skip)
    dec2_with_skip = Concatenate()([dec2, enc1])
    u_net_style_model = Model(inputs=inputs, outputs=dec2_with_skip)


    
    # u-net style with convolutional 1D 
    inputs = Input(shape=(256, 1))                                              # 1D sequence with 256 timesteps
    enc1 = Conv1D(64, 3, padding='same', activation='relu')(inputs)             # Encoder
    enc2 = Conv1D(128, 3, padding='same', activation='relu')(enc1)
    pool1 = MaxPooling1D(2)(enc2)
    bottleneck = Conv1D(256, 3, padding='same', activation='relu')(pool1)       # Bottleneck
    up1 = UpSampling1D(2)(bottleneck)                                           # Decoder
    dec1 = Conv1D(128, 3, padding='same', activation='relu')(up1)
    dec1_skip = Concatenate()([dec1, enc2])                                     # Skip connection
    dec2 = Conv1D(64, 3, padding='same', activation='relu')(dec1_skip)
    dec2_skip = Concatenate()([dec2, enc1])                                     # Skip connection
    x = GlobalAveragePooling1D()(dec2_skip)                                     # Flatten sequence
    u_net_model = Model(inputs=inputs, outputs=x)


    # multi-scale processing (inception-style)
    inputs = Input(shape=(256,))
    branch1 = Dense(32, activation='relu')(inputs)                          # Branch 1: 1x1 convolution equivalent
    branch2 = Dense(64, activation='relu')(inputs)                          # Branch 2: 3x3 equivalent (wider network)
    branch2 = Dense(32, activation='relu')(branch2)
    branch3 = Dense(96, activation='relu')(inputs)                          # Branch 3: 5x5 equivalent (even wider)
    branch3 = Dense(64, activation='relu')(branch3)
    branch3 = Dense(32, activation='relu')(branch3)
    branch4 = Dense(32, activation='relu')(inputs)                          # Branch 4: Pooling equivalent
    merged = Concatenate()([branch1, branch2, branch3, branch4])            # Concatenate all branches
    multi_scale_model = Model(inputs=inputs, outputs=merged)



    # multi-scale / inception-style (convolutional 1D)
    inputs = Input(shape=(256, 1))                                                  # 1D sequence
    branch1 = Conv1D(32, 1, padding='same', activation='relu')(inputs)              # Branch 1: 1x1 convolution
    branch2 = Conv1D(32, 3, padding='same', activation='relu')(inputs)              # Branch 2: 3x1 convolution
    branch2 = Conv1D(32, 3, padding='same', activation='relu')(branch2)
    branch3 = Conv1D(32, 5, padding='same', activation='relu')(inputs)              # Branch 3: 5x1 convolution
    branch3 = Conv1D(32, 5, padding='same', activation='relu')(branch3)
    branch4 = MaxPooling1D(3, strides=1, padding='same')(inputs)                    # Branch 4: Max pooling
    branch4 = Conv1D(32, 1, padding='same', activation='relu')(branch4)
    merged = Concatenate()([branch1, branch2, branch3, branch4])                    # Merge all branches
    x = GlobalAveragePooling1D()(merged)                                            
    inception_model = Model(inputs=inputs, outputs=x)



    # custom layer operation
    inputs = Input(shape=(100,))
    custom = Lambda(lambda x: tf.square(x))(inputs)
    lambda_model = Model(inputs=inputs, outputs=custom)



if __name__ == '__main__':
    main()