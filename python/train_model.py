#!/usr/bin/env python3
"""
Train the E-Nose AI model using TensorFlow
Implements the Compressed CNN architecture from the paper
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os

def create_ccnn_model(input_shape, num_classes):
    """
    Create the Compressed CNN model as described in the paper
    Total parameters: ~15,000
    """
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=input_shape),
        
        # Reshape for 1D convolution (sensors × features)
        keras.layers.Reshape((input_shape[0], 1)),
        
        # Conv1D layer 1: 16 filters, kernel size 3
        keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),
        
        # Conv1D layer 2: 32 filters, kernel size 3
        keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        
        # Global Average Pooling
        keras.layers.GlobalAveragePooling1D(),
        
        # Dense layers
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        
        # Output layer
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_and_preprocess_data(data_path):
    """
    Load and preprocess sensor data
    Expected format: CSV with columns for sensor readings and labels
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Assume last column is the label
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Extract features (6 sensors × 10 features each = 60 features)
    # For now, use raw values and compute basic features
    features = extract_features(X)
    
    # Convert labels to categorical
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    return features, y_categorical, label_encoder.classes_

def extract_features(raw_data):
    """
    Extract features from raw sensor data
    Features per sensor:
    1. Mean value
    2. Standard deviation
    3. Max value
    4. Min value
    5. Peak-to-peak amplitude
    6. Response time (simplified)
    7. Recovery time (simplified)
    8. Area under curve
    9. Slope
    10. Frequency domain feature (simplified)
    """
    n_samples = raw_data.shape[0]
    n_sensors = 6
    n_features = 10
    
    features = np.zeros((n_samples, n_sensors * n_features))
    
    for i in range(n_samples):
        sample = raw_data[i]
        
        # Reshape if needed (assuming 6 sensors)
        if len(sample) > n_sensors:
            # If we have time series data, reshape
            sample = sample.reshape(n_sensors, -1)
            
            for s in range(n_sensors):
                sensor_data = sample[s]
                
                # Calculate features
                feat_idx = s * n_features
                features[i, feat_idx] = np.mean(sensor_data)      # Mean
                features[i, feat_idx + 1] = np.std(sensor_data)   # Std
                features[i, feat_idx + 2] = np.max(sensor_data)   # Max
                features[i, feat_idx + 3] = np.min(sensor_data)   # Min
                features[i, feat_idx + 4] = np.ptp(sensor_data)   # Peak-to-peak
                features[i, feat_idx + 5] = np.argmax(sensor_data) # Response time
                features[i, feat_idx + 6] = len(sensor_data) - np.argmax(sensor_data[::-1]) # Recovery
                features[i, feat_idx + 7] = np.trapz(sensor_data) # Area under curve
                features[i, feat_idx + 8] = np.gradient(sensor_data).mean() # Slope
                features[i, feat_idx + 9] = np.abs(np.fft.fft(sensor_data)[1]) # FFT feature
        else:
            # If we have pre-computed features, use them directly
            features[i] = sample[:n_sensors * n_features]
    
    return features

def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    """
    Train the model with early stopping and learning rate reduction
    """
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def convert_to_tflite(model, save_path='models/quantized_model.tflite'):
    """
    Convert and quantize model for edge deployment
    """
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Quantize to int8
    converter.target_spec.supported_types = [tf.int8]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print model size
    size_kb = len(tflite_model) / 1024
    print(f"Quantized model size: {size_kb:.1f} KB")
    
    return tflite_model

def main():
    parser = argparse.ArgumentParser(description='Train E-Nose AI Model')
    parser.add_argument('--dataset', type=str, default='data/food_quality.csv',
                        help='Path to training dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--split', type=float, default=0.2,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    print("🔬 E-Nose AI Model Training")
    print("=" * 50)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    X, y, classes = load_and_preprocess_data(args.dataset)
    print(f"Loaded {len(X)} samples with {len(classes)} classes")
    print(f"Classes: {classes}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.split, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Create model
    print("\nCreating CCNN model...")
    model = create_ccnn_model(
        input_shape=(X_train.shape[1],),
        num_classes=y.shape[1]
    )
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, X_val, y_val, args.epochs)
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {accuracy:.1%}")
    
    # Save models
    print("\nSaving models...")
    model.save('models/base_model.h5')
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    convert_to_tflite(model)
    
    print("\n✅ Training complete!")
    print(f"Models saved in: models/")

if __name__ == "__main__":
    # If no dataset exists, create a sample one
    if not os.path.exists('data/food_quality.csv'):
        print("Creating sample dataset...")
        os.makedirs('data', exist_ok=True)
        
        # Create synthetic data for demonstration
        n_samples = 1000
        n_sensors = 6
        
        # Generate synthetic sensor readings
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # Randomly choose a class
            class_idx = np.random.randint(0, 5)
            classes = ['Fresh', 'Slightly_Spoiled', 'Spoiled', 'Fermented', 'Contaminated']
            
            # Generate sensor readings based on class
            base_values = {
                'Fresh': [0.2, 0.3, 0.25, 0.28, 0.22, 0.26],
                'Slightly_Spoiled': [0.4, 0.5, 0.45, 0.48, 0.42, 0.46],
                'Spoiled': [0.7, 0.8, 0.75, 0.78, 0.72, 0.76],
                'Fermented': [0.5, 0.6, 0.55, 0.58, 0.52, 0.56],
                'Contaminated': [0.9, 0.85, 0.88, 0.92, 0.86, 0.90]
            }
            
            class_name = classes[class_idx]
            base = base_values[class_name]
            
            # Add noise
            readings = [b + np.random.normal(0, 0.05) for b in base]
            readings.append(class_name)
            
            data.append(readings)
        
        # Save as CSV
        columns = [f'MQ{i}' for i in [2, 3, 4, 5, 7, 135]] + ['Label']
        df = pd.DataFrame(data, columns=columns)
        df.to_csv('data/food_quality.csv', index=False)
        print("Sample dataset created: data/food_quality.csv")
    
    # Run main training
    main()