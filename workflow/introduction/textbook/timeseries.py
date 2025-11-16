"""
@author: Ananda Francis
@file: timeseries.py: Familiarize myself with the code from Deep Learning with Python Textbook (pg 289 - ?)
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



def main():
    # intialize data for model development
    data = np.load("../../../datasets/spotify_dataset.npz", allow_pickle=True)
    _, class_count = data['train_labels'].shape
    feature_shape = data['train_features'].shape[1:]

    print(f"train set: {data['train_features'].shape}")
    print(f"validation set: {data['val_features'].shape}")
    print(f"test set: {data['test_features'].shape}")
    print(f'class count: {class_count}, feature shape: {feature_shape}\n\n')


    # densely connected multi layer perceptron
    inputs = keras.Input(shape=(feature_shape))
    x = layers.Flatten()(inputs)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(class_count, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("jena_dense.keras",
    #                                     save_best_only=True)
    # ]
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    history = model.fit(
        data['train_features'], 
        data['train_labels'],
        epochs=10,
        # validation_data=val_dataset,
        # callbacks=callbacks
    )
    print('MLP history: ', history.history)
    
    # 1D convolutional neural network
    inputs = keras.Input(shape=(feature_shape))
    x = layers.Conv1D(8, 24, activation="relu")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(8, 12, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(8, 6, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(class_count, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("jena_conv.keras",
    #                                     save_best_only=True)
    # ]
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    history = model.fit(
        data['train_features'], 
        data['train_labels'],
        epochs=10,
        # validation_data=val_dataset,
        # callbacks=callbacks
    )
    print('CNN history: ', history.history)

    # simple long short-term memory recurrent neural network
    inputs = keras.Input(shape=(feature_shape))
    x = layers.LSTM(16)(inputs)
    outputs = layers.Dense(class_count, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("jena_lstm.keras",
    #                                     save_best_only=True)
    # ]
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    history = model.fit(
        data['train_features'], 
        data['train_labels'],
        epochs=10,
        # validation_data=val_dataset,
        # callbacks=callbacks
    )
    print('LSTM history: ', history.history)


    
if __name__ == '__main__':
      main()