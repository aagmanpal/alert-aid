"""
LSTM Flood Prediction Model
===========================
Time-series deep learning model for river level forecasting.

This module implements a real LSTM neural network using TensorFlow/Keras
for predicting river water levels based on historical rainfall and level data.

Architecture:
- Input: 7-day sequence of (rainfall, river_level)
- LSTM layers with dropout for regularization
- Dense output for next-day river level prediction
"""

import numpy as np
import os
import json
from typing import Tuple, Dict, Optional, List
from datetime import datetime

# Try importing TensorFlow, fall back to simulation if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. Using simulation mode.")

# Import sklearn for scaling
try:
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LSTMFloodModel:
    """
    LSTM-based flood prediction model for time-series river level forecasting.
    
    The model learns temporal patterns in rainfall and river level data
    to predict future water levels with 24-48 hour lead time.
    """
    
    def __init__(
        self,
        sequence_length: int = 7,
        n_features: int = 2,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        model_path: Optional[str] = None
    ):
        """
        Initialize LSTM flood model.
        
        Args:
            sequence_length: Number of time steps in input sequence (days)
            n_features: Number of input features (rainfall, river_level)
            lstm_units: Units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            model_path: Path to load pre-trained model
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.training_history = None
        self.model_metadata = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif TF_AVAILABLE:
            self._build_model()
    
    def _build_model(self) -> None:
        """Build the LSTM neural network architecture."""
        if not TF_AVAILABLE:
            return
            
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # First LSTM layer with return sequences
            layers.LSTM(
                self.lstm_units[0],
                return_sequences=True if len(self.lstm_units) > 1 else False,
                kernel_regularizer=keras.regularizers.l2(0.01)
            ),
            layers.Dropout(self.dropout_rate),
            layers.BatchNormalization(),
        ])
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            model.add(layers.LSTM(
                units,
                return_sequences=return_seq,
                kernel_regularizer=keras.regularizers.l2(0.01)
            ))
            model.add(layers.Dropout(self.dropout_rate))
            model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate / 2))
        
        # Output layer (single value: next day river level)
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print("✅ LSTM model built successfully")
        print(f"   Architecture: Input({self.sequence_length}x{self.n_features}) → "
              f"LSTM{self.lstm_units} → Dense(16) → Output(1)")
    
    def prepare_sequences(
        self,
        rainfall: np.ndarray,
        river_level: np.ndarray,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for LSTM training/prediction.
        
        Args:
            rainfall: Array of daily rainfall values
            river_level: Array of daily river level values
            fit_scaler: Whether to fit the scaler (True for training)
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        # Stack features
        features = np.column_stack([rainfall, river_level])
        
        # Scale features
        if self.scaler and fit_scaler:
            features = self.scaler.fit_transform(features)
        elif self.scaler:
            features = self.scaler.transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features) - 1):
            X.append(features[i - self.sequence_length:i])
            y.append(features[i + 1, 1])  # Next day river level (scaled)
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        model_save_path: Optional[str] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_save_path: Path to save best model
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if not TF_AVAILABLE or self.model is None:
            print("⚠️ TensorFlow not available. Returning simulated training.")
            return self._simulate_training()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(
                model_save_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            ))
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Split for validation if not provided
        if validation_data is None:
            split_idx = int(len(X_train) * 0.9)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
            validation_data = (X_val, y_val)
        
        print(f"🏋️ Training LSTM model...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        # Calculate metrics
        train_metrics = self.model.evaluate(X_train, y_train, verbose=0)
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        
        self.model_metadata = {
            "trained_at": datetime.now().isoformat(),
            "epochs_completed": len(history.history['loss']),
            "train_loss": float(train_metrics[0]),
            "train_mae": float(train_metrics[1]),
            "val_loss": float(val_metrics[0]),
            "val_mae": float(val_metrics[1]),
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "lstm_units": self.lstm_units,
        }
        
        print(f"✅ Training complete!")
        print(f"   Train MAE: {train_metrics[1]:.4f}")
        print(f"   Val MAE: {val_metrics[1]:.4f}")
        
        return self.training_history
    
    def _simulate_training(self) -> Dict:
        """Simulate training when TensorFlow is not available."""
        self.is_trained = True
        self.model_metadata = {
            "trained_at": datetime.now().isoformat(),
            "mode": "simulated",
            "simulated_mae": 2.5,
        }
        return {"loss": [0.1], "val_loss": [0.12]}
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if not TF_AVAILABLE or self.model is None:
            return self._simulate_prediction(X)
        
        # Get predictions
        predictions_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform to get actual river levels
        if self.scaler:
            # Create dummy array for inverse transform
            dummy = np.zeros((len(predictions_scaled), 2))
            dummy[:, 1] = predictions_scaled.flatten()
            predictions = self.scaler.inverse_transform(dummy)[:, 1]
        else:
            predictions = predictions_scaled.flatten()
        
        # Calculate confidence (based on prediction variance using Monte Carlo Dropout)
        confidence = None
        if return_confidence:
            confidence = self._calculate_confidence(X)
        
        return predictions, confidence
    
    def _simulate_prediction(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate predictions when model is not available."""
        # Use last value in sequence with some noise
        last_values = X[:, -1, 1] if len(X.shape) == 3 else X[-1, 1]
        predictions = last_values * (1 + np.random.normal(0, 0.05, size=len(X) if len(X.shape) == 3 else 1))
        confidence = np.ones_like(predictions) * 0.75
        return predictions, confidence
    
    def _calculate_confidence(
        self,
        X: np.ndarray,
        n_samples: int = 10
    ) -> np.ndarray:
        """
        Calculate prediction confidence using Monte Carlo Dropout.
        
        Args:
            X: Input sequences
            n_samples: Number of forward passes
            
        Returns:
            Confidence scores (0-1)
        """
        if not TF_AVAILABLE or self.model is None:
            return np.ones(len(X)) * 0.8
        
        # Multiple forward passes with dropout enabled
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)  # Enable dropout
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate variance
        variance = np.var(predictions, axis=0).flatten()
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = 1 / (1 + variance)
        confidence = np.clip(confidence, 0.5, 0.95)
        
        return confidence
    
    def predict_multi_horizon(
        self,
        initial_sequence: np.ndarray,
        forecast_rainfall: np.ndarray,
        horizons: List[int] = [1, 2, 3]
    ) -> Dict:
        """
        Predict river levels for multiple future horizons.
        
        Args:
            initial_sequence: Last sequence_length days of data
            forecast_rainfall: Forecasted rainfall for future days
            horizons: Days ahead to predict [1, 2, 3] = tomorrow, day after, etc.
            
        Returns:
            Dictionary with predictions for each horizon
        """
        results = {}
        current_sequence = initial_sequence.copy()
        
        for horizon in sorted(horizons):
            # Make prediction
            if len(current_sequence.shape) == 2:
                X = current_sequence.reshape(1, *current_sequence.shape)
            else:
                X = current_sequence
                
            pred, conf = self.predict(X)
            
            results[f"{horizon}d"] = {
                "predicted_level": float(pred[0]),
                "confidence": float(conf[0]) if conf is not None else 0.8,
                "forecast_rainfall": float(forecast_rainfall[horizon - 1]) if horizon <= len(forecast_rainfall) else 0
            }
            
            # Update sequence for next prediction (auto-regressive)
            if horizon < max(horizons) and horizon <= len(forecast_rainfall):
                new_entry = np.array([[forecast_rainfall[horizon - 1], pred[0]]])
                if self.scaler:
                    new_entry = self.scaler.transform(new_entry)
                current_sequence = np.vstack([current_sequence[1:], new_entry])
        
        return results
    
    def save_model(self, path: str) -> None:
        """Save model and associated files."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        if TF_AVAILABLE and self.model:
            self.model.save(path)
            print(f"✅ Model saved to {path}")
        
        # Save scaler
        if self.scaler:
            scaler_path = path.replace('.h5', '_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        meta_path = path.replace('.h5', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
    
    def load_model(self, path: str) -> None:
        """Load pre-trained model."""
        if TF_AVAILABLE and os.path.exists(path):
            self.model = keras.models.load_model(path)
            self.is_trained = True
            print(f"✅ Model loaded from {path}")
        
        # Load scaler
        scaler_path = path.replace('.h5', '_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        meta_path = path.replace('.h5', '_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.model_metadata = json.load(f)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if TF_AVAILABLE and self.model:
            import io
            stream = io.StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            return stream.getvalue()
        return "Model not available (TensorFlow not installed)"


class FloodLevelPredictor:
    """
    High-level interface for flood prediction combining LSTM with business logic.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        danger_level: float = 100.0,
        warning_level: float = 85.0
    ):
        """
        Initialize flood predictor.
        
        Args:
            model_path: Path to pre-trained LSTM model
            danger_level: River level threshold for flood (meters)
            warning_level: River level threshold for warning (meters)
        """
        self.lstm_model = LSTMFloodModel(model_path=model_path)
        self.danger_level = danger_level
        self.warning_level = warning_level
    
    def predict_flood_risk(
        self,
        recent_rainfall: List[float],
        recent_levels: List[float],
        forecast_rainfall: List[float]
    ) -> Dict:
        """
        Predict flood risk for the next few days.
        
        Args:
            recent_rainfall: Last 7 days rainfall (mm)
            recent_levels: Last 7 days river levels (m)
            forecast_rainfall: Next 3 days forecasted rainfall (mm)
            
        Returns:
            Comprehensive flood risk assessment
        """
        # Prepare input sequence
        sequence = np.column_stack([recent_rainfall, recent_levels])
        
        # Scale if scaler available
        if self.lstm_model.scaler:
            sequence = self.lstm_model.scaler.transform(sequence)
        
        # Get multi-horizon predictions
        predictions = self.lstm_model.predict_multi_horizon(
            sequence,
            forecast_rainfall,
            horizons=[1, 2, 3]
        )
        
        # Analyze risk
        risk_assessment = {
            "predictions": predictions,
            "current_level": recent_levels[-1],
            "risk_levels": {},
            "alerts": [],
            "recommendation": "",
            "pattern_analysis": self._analyze_patterns(recent_rainfall, recent_levels)
        }
        
        # Determine risk for each horizon
        max_risk = "LOW"
        for horizon, pred in predictions.items():
            level = pred["predicted_level"]
            
            if level >= self.danger_level:
                risk = "CRITICAL"
            elif level >= self.warning_level:
                risk = "HIGH"
            elif level >= self.warning_level * 0.8:
                risk = "MODERATE"
            else:
                risk = "LOW"
            
            risk_assessment["risk_levels"][horizon] = {
                "level": level,
                "risk": risk,
                "above_danger": level >= self.danger_level,
                "above_warning": level >= self.warning_level,
            }
            
            if risk == "CRITICAL" or (risk == "HIGH" and max_risk not in ["CRITICAL"]):
                max_risk = risk
            elif risk == "MODERATE" and max_risk == "LOW":
                max_risk = risk
        
        risk_assessment["overall_risk"] = max_risk
        
        # Generate alerts
        if max_risk == "CRITICAL":
            risk_assessment["alerts"].append({
                "type": "FLOOD_WARNING",
                "severity": "CRITICAL",
                "message": f"⚠️ FLOOD LIKELY: River predicted to reach {predictions['1d']['predicted_level']:.1f}m (danger: {self.danger_level}m)"
            })
            risk_assessment["recommendation"] = "EVACUATE immediately to higher ground. Follow official evacuation routes."
        elif max_risk == "HIGH":
            risk_assessment["alerts"].append({
                "type": "FLOOD_WATCH",
                "severity": "HIGH",
                "message": f"🔶 Flood Watch: River approaching danger level. Monitor closely."
            })
            risk_assessment["recommendation"] = "Prepare for possible evacuation. Move valuables to higher floors."
        elif max_risk == "MODERATE":
            risk_assessment["alerts"].append({
                "type": "ADVISORY",
                "severity": "MODERATE",
                "message": "🔷 Advisory: Elevated water levels expected. Stay informed."
            })
            risk_assessment["recommendation"] = "Stay alert for updates. Avoid flood-prone areas."
        else:
            risk_assessment["recommendation"] = "No immediate flood risk. Continue normal activities."
        
        return risk_assessment
    
    def _analyze_patterns(
        self,
        rainfall: List[float],
        levels: List[float]
    ) -> Dict:
        """Analyze rainfall and level patterns."""
        rainfall_arr = np.array(rainfall)
        
        return {
            "total_rainfall_7d": float(np.sum(rainfall_arr)),
            "avg_rainfall_7d": float(np.mean(rainfall_arr)),
            "max_rainfall_day": float(np.max(rainfall_arr)),
            "rainfall_trend": "increasing" if rainfall_arr[-3:].mean() > rainfall_arr[:3].mean() else "decreasing",
            "level_trend": "rising" if levels[-1] > levels[0] else "falling",
            "heavy_rain_days": int(np.sum(rainfall_arr > 50)),
            "pattern_note": self._get_pattern_note(rainfall_arr)
        }
    
    def _get_pattern_note(self, rainfall: np.ndarray) -> str:
        """Generate human-readable pattern note."""
        total = np.sum(rainfall)
        heavy_days = np.sum(rainfall > 50)
        
        if total > 200 and heavy_days >= 3:
            return "⚠️ Heavy sustained rainfall detected (>200mm over 3+ days) - historically associated with flooding"
        elif np.max(rainfall) > 100:
            return "⚠️ Extreme rainfall event detected - monitor river levels closely"
        elif total > 150:
            return "🔶 Elevated rainfall levels - some risk of rising water"
        else:
            return "✅ Normal rainfall patterns - low flood risk"


if __name__ == "__main__":
    # Test the LSTM model
    print("🧠 Testing LSTM Flood Model")
    print("=" * 50)
    
    # Initialize model
    model = LSTMFloodModel(sequence_length=7, n_features=2)
    
    # Generate synthetic test data
    np.random.seed(42)
    n_days = 200
    rainfall = np.random.gamma(2, 30, n_days)
    river_level = np.zeros(n_days)
    river_level[0] = 50
    
    for i in range(1, n_days):
        river_level[i] = 0.8 * river_level[i-1] + 0.2 * rainfall[i] - 5 + np.random.normal(0, 1)
        river_level[i] = max(30, river_level[i])
    
    # Prepare sequences
    X, y = model.prepare_sequences(rainfall, river_level)
    print(f"Prepared {len(X)} sequences")
    
    # Train (if TensorFlow available)
    if TF_AVAILABLE:
        history = model.train(X, y, epochs=20, verbose=1)
        
        # Test prediction
        test_pred, conf = model.predict(X[-10:])
        print(f"\nSample predictions vs actual:")
        for i in range(5):
            print(f"  Predicted: {test_pred[i]:.2f}m, Actual: {y[-10+i]:.2f}m, Confidence: {conf[i]:.2f}")
    else:
        print("⚠️ TensorFlow not available - using simulation mode")
        
    print("\n✅ LSTM model test complete!")
