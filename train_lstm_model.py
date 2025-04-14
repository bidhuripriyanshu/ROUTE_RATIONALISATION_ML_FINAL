import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import seaborn as sns

class TrafficPredictor:
    def __init__(self, sequence_length=10, prediction_horizon=5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.prediction_horizon)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_data(self, data, target_col='occupancy'):
       
        scaled_data = self.scaler.fit_transform(data[target_col].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon + 1):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def plot_training_history(self, history, X_test, y_test):
        # Create a directory for visualizations if it doesn't exist
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
        
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training History
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss During Training', fontsize=14, pad=20)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. MAE History
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
        ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_title('Model MAE During Training', fontsize=14, pad=20)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Mean Absolute Error', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Loss Distribution
        ax3 = plt.subplot(2, 2, 3)
        sns.histplot(history.history['loss'], bins=50, label='Training Loss', alpha=0.5)
        sns.histplot(history.history['val_loss'], bins=50, label='Validation Loss', alpha=0.5)
        ax3.set_title('Loss Distribution', fontsize=14, pad=20)
        ax3.set_xlabel('Loss Value', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Final Predictions vs Actual
        y_pred = self.model.predict(X_test)
        
        # Reshape arrays to 2D for inverse_transform
        y_test_reshaped = y_test.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)
        
        y_test_inv = self.scaler.inverse_transform(y_test_reshaped)
        y_pred_inv = self.scaler.inverse_transform(y_pred_reshaped)
        
        ax4 = plt.subplot(2, 2, 4)
        ax4.scatter(y_test_inv, y_pred_inv, alpha=0.5)
        min_val = min(y_test_inv.min(), y_pred_inv.min())
        max_val = max(y_test_inv.max(), y_pred_inv.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax4.set_title('Predicted vs Actual Values', fontsize=14, pad=20)
        ax4.set_xlabel('Actual Values', fontsize=12)
        ax4.set_ylabel('Predicted Values', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle('LSTM Traffic Prediction Model Analysis', fontsize=16, y=0.95)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('visualizations/training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual plots
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss During Training', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/loss_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, data_file, epochs=500, batch_size=32, validation_split=0.2):
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        
        # Group by timestamp and calculate average occupancy
        df_grouped = df.groupby('timestamp')['occupancy'].mean().reset_index()
        
        # Prepare data
        X, y = self.prepare_data(df_grouped)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        self.model = self.build_model(input_shape=(self.sequence_length, 1))
        
        # Create checkpoint callback
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Train the model
        print("\nTraining LSTM model...")
        print(f"Total epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_test)}")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint],
            verbose=1
        )
        
        # Save scaler parameters
        scaler_params = {
            'scale_': self.scaler.scale_.tolist(),
            'min_': self.scaler.min_.tolist()
        }
        with open('scaler_params.json', 'w') as f:
            json.dump(scaler_params, f)
        
        # Plot training history and analysis
        self.plot_training_history(history, X_test, y_test)
        
        # Evaluate model
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nFinal Test Loss: {test_loss:.4f}")
        print(f"Final Test MAE: {test_mae:.4f}")
        
        return history
    
    def predict(self, data):
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        X = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        prediction = self.model.predict(X)
        return self.scaler.inverse_transform(prediction)[0]

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_lstm_model.py <traffic_data_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    predictor = TrafficPredictor(sequence_length=10, prediction_horizon=5)
    predictor.train(data_file)

if __name__ == "__main__":
    main() 