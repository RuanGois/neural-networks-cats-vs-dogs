import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def mlp(input_shape=(128, 128, 3), num_classes=2):
    """
    Modelo MLP Baseline para Cats vs Dogs
    
    Args:
        input_shape: Forma da imagem (ex: (128, 128, 3))
        num_classes: Número de classes (2 para cats vs dogs)
    """
    model = models.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=input_shape),
        
        # Camada 1
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Camada 2
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Camada 3
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Camada 4
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Saída
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
   
    return model

def cnn(input_shape=(128, 128, 3), num_classes=2):
    """
    Modelo CNN para Cats vs Dogs
    """
    model = tf.keras.Sequential([
        # Bloco 1
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Bloco 2
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Bloco 3
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Classificador
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

if __name__ == "__main__":
    # Teste MLP
    mlp_model = mlp(input_shape=(128, 128, 3))
    mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("MLP Summary:")
    mlp_model.summary()
    
    # Teste CNN
    cnn_model = cnn(input_shape=(128, 128, 3))
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("\nCNN Summary:")
    cnn_model.summary()
    