import tensorflow as tf
import tensorflow_datasets as tfds

def download_prepare():
    """Download e preparação do dataset Cats vs Dogs"""

    # Download:
    (raw_train, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True,
        shuffle_files=True
    )

    return raw_train, raw_test, metadata

def preprocess_image(image, label, img_size=(224, 224)):
    """Pré-processamento básico das imagens"""

    # Redimensionar
    image = tf.image.resize(image, img_size)

    # Normaliza para [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

def get_datasets(batch_size=32, img_size=(224, 224)):
    """Retorna datasets pré-processados"""
    raw_train, raw_test, metadata = download_prepare()

    # Aplica pré-processamento
    train_dataset = raw_train.map(
        lambda x, y: preprocess_image(x, y, img_size)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


    test_dataset = raw_test.map(
        lambda x, y: preprocess_image(x, y, img_size)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, metadata

# Teste
if __name__ == "__main__":
    train_ds, test_ds, metadata = get_datasets()
    print(f"Classes: {metadata.features['label'].names}")
    print(f"Train batches: {len(list(train_ds))}")
    print(f"Test batches: {len(list(test_ds))}")