import tensorflow as tf
from models import mlp, cnn
from data import get_datasets
import matplotlib.pyplot as plt
import numpy as np
import os

def train_model(model_type='mlp', epochs=50, batch_size=32, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Carregar dados
    train_ds, test_ds, metadata = get_datasets(batch_size=batch_size)
    
    # Criar modelo
    if model_type == 'mlp':
        save_path = "../models/mlp_model.h5"
        model = mlp(input_shape=(224, 224, 3), num_classes=2)
    elif model_type == 'cnn':
        save_path = "../models/cnn_model.h5"
        model = cnn(input_shape=(224, 224, 3), num_classes=2)
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss',
                                           save_best_only=True, save_weights_only=False,
                                           verbose=1)
    ]
    
    # Treinar
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks
    )
    
    return model, history
