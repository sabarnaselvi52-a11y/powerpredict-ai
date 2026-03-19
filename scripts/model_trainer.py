import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Feature-Target separation
    X = df.drop(columns=['actual_consumption'])
    y = df['actual_consumption']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalization/Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use in prediction
    joblib.dump(scaler, r'C:\Users\velus\.gemini\antigravity\scratch\electricity_prediction\models\scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def build_ann_model(input_shape):
    model = models.Sequential([
        # Input layer and 1st hidden layer
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        # 2nd hidden layer
        layers.Dense(32, activation='relu'),
        # Output layer
        layers.Dense(1) # Linear activation for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate():
    data_path = r'C:\Users\velus\.gemini\antigravity\scratch\electricity_prediction\data\electricity_data.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    model = build_ann_model(X_train.shape[1])
    
    # Training
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluation
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.2f}")
    
    # Save model
    model_save_path = r'C:\Users\velus\.gemini\antigravity\scratch\electricity_prediction\models\electricity_ann_model.keras'
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plots
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    # Predictions vs Actual
    y_pred = model.predict(X_test).flatten()
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Predicted vs Actual Consumption')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.tight_layout()
    plot_path = r'C:\Users\velus\.gemini\antigravity\scratch\electricity_prediction\data\training_plots.png'
    plt.savefig(plot_path)
    print(f"Training plots saved to {plot_path}")

if __name__ == "__main__":
    train_and_evaluate()
