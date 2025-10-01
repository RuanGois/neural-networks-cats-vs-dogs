import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def MLP(input_shape, hidden_units=[128, 64], num_classes=2, dropout_rate=0.3, l2_reg=0.001):
    """
    Modelo MLP Baseline para Cats vs Dogs
    
    Args:
        input_shape: Forma da imagem (ex: (150, 150, 3))
        hidden_units: Lista com número de neurônios nas camadas ocultas
        num_classes: Número de classes (2 para cats vs dogs)
        dropout_rate: Taxa de dropout para regularização
        l2_reg: Força da regularização L2
    """
    model = models.Sequential()
    
    # Camada de entrada
    model.add(layers.GlobalAveragePooling2D(input_shape=input_shape))

    # Camadas ocultas
    for units in hidden_units:
        model.add(layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)  # Regularização L2
        ))
        model.add(layers.Dropout(dropout_rate))  # Regularização Dropout
    
    # Camada de saída
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def CNN(input_shape=(150, 150, 3), num_classes=2, dropout_rate=0.5, l2_reg=0.001):
    """
    Modelo CNN para Cats vs Dogs
    """
    model = models.Sequential([
        # Primeiro bloco convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                      kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),

        # Segundo bloco convolucional
        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),

        # Terceiro bloco convolucional
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D((2, 2)),

        # Classificador
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

if __name__ == "__main__":
    # Teste MLP
    mlp_model = MLP(input_shape=(224, 224, 3))
    mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("MLP Summary:")
    mlp_model.summary()
    
    # Teste CNN
    cnn_model = CNN(input_shape=(224, 224, 3))
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("\nCNN Summary:")
    cnn_model.summary()
    