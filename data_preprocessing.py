import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32

def load_data(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    return train_generator, test_generator

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_data(base_dir)