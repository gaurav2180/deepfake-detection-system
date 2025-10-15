import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def swish_activation(x):
    return x * tf.nn.sigmoid(x)

def create_resnet_swish_bilstm():
    # Input layer for 224x224x3 images
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Basic ResNet18 backbone (simplified for now)
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Lambda(swish_activation)(x)  # Swish activation
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks (simplified)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Lambda(swish_activation)(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Reshape for LSTM
    x = layers.Reshape((1, -1))(x)
    
    # BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(500, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(200))(x)
    
    # Final classification
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(2, activation='softmax')(x)  # Real vs Fake
    
    model = keras.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = create_resnet_swish_bilstm()
    model.summary()
    print("Model created successfully!")
