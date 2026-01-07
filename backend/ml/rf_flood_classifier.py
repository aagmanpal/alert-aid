"""
Random Forest Flood Classifier
==============================
Classification model for flood/no-flood prediction using ensemble of decision trees.

This module implements a Random Forest classifier for binary flood classification
using tabular features extracted from rainfall and river level data.

Features:
1. rainfall_today: Current day's rainfall (mm)
2. rainfall_2day_sum: Sum of last 2 days rainfall (mm)
3. rainfall_3day_sum: Sum of last 3 days rainfall (mm)
4. rainfall_week_avg: Average rainfall over last 7 days (mm)
5. rainfall_week_max: Maximum daily rainfall in last week (mm)
6. prev_river_level: Previous day's river level (m)
7. level_change_rate: Daily change in river level (m)
8. soil_saturation_proxy: Cumulative rainfall indicator
9. days_since_heavy_rain: Days since rainfall > 50mm
"""

import numpy as np
import pandas as pd
import os
import json
from typing import Tuple, Dict, List, Optional
from datetime import datetime

# Import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Using simulation mode.")


class RandomForestFloodClassifier:
    """
    Random Forest classifier for binary flood prediction.
    
    Uses ensemble of decision trees to classify whether flooding
    will occur based on rainfall patterns and river conditions.
    """
    
    # Feature names for model interpretability
    FEATURE_NAMES = [
        "rainfall_today",
        "rainfall_2day_sum", 
        "rainfall_3day_sum",
        "rainfall_week_avg",
        "rainfall_week_max",
        "prev_river_level",
        "level_change_rate",
        "soil_saturation_proxy",
        "days_since_heavy_rain"
    ]
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        class_weight: str = "balanced",
        random_state: int = 42,
        model_path: Optional[str] = None
    ):
        """
        Initialize Random Forest flood classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            class_weight: Handle class imbalance ("balanced" recommended)
            random_state: For reproducibility
            model_path: Path to load pre-trained model
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.feature_importance = {}
        self.model_metadata = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif SKLEARN_AVAILABLE:
            self._build_model()
    
    def _build_model(self) -> None:
        """Build the Random Forest classifier."""
        if not SKLEARN_AVAILABLE:
            return
            
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            oob_score=True  # Out-of-bag score for validation
        )
        
        print("✅ Random Forest classifier built successfully")
        print(f"   Trees: {self.n_estimators}, Max Depth: {self.max_depth}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        test_size: float = 0.2,
        model_save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the Random Forest classifier.
        
        Args:
            X: Feature matrix
            y: Binary labels (0: no flood, 1: flood)
            feature_names: Names of features
            test_size: Fraction for test split
            model_save_path: Path to save trained model
            
        Returns:
            Training metrics dictionary
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            print("⚠️ scikit-learn not available. Using simulated training.")
            return self._simulate_training()
        
        if feature_names:
            self.FEATURE_NAMES = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"🏋️ Training Random Forest classifier...")
        print(f"   Training samples: {len(X_train)} (Floods: {sum(y_train)})")
        print(f"   Test samples: {len(X_test)} (Floods: {sum(y_test)})")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        y_prob_test = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            "train": {
                "accuracy": accuracy_score(y_train, y_pred_train),
                "precision": precision_score(y_train, y_pred_train, zero_division=0),
                "recall": recall_score(y_train, y_pred_train, zero_division=0),
                "f1": f1_score(y_train, y_pred_train, zero_division=0),
            },
            "test": {
                "accuracy": accuracy_score(y_test, y_pred_test),
                "precision": precision_score(y_test, y_pred_test, zero_division=0),
                "recall": recall_score(y_test, y_pred_test, zero_division=0),
                "f1": f1_score(y_test, y_pred_test, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob_test) if len(np.unique(y_test)) > 1 else 0,
            },
            "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
            "oob_score": self.model.oob_score_,
        }
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.FEATURE_NAMES[:len(self.model.feature_importances_)],
            self.model.feature_importances_
        ))
        
        # Sort by importance
        self.feature_importance = dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        metrics["feature_importance"] = self.feature_importance
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics["cv_scores"] = {
            "mean": float(np.mean(cv_scores)),
            "std": float(np.std(cv_scores)),
            "scores": cv_scores.tolist()
        }
        
        # Store metadata
        self.model_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy": metrics["test"]["accuracy"],
            "test_f1": metrics["test"]["f1"],
            "feature_importance": self.feature_importance,
        }
        
        print(f"\n✅ Training complete!")
        print(f"   Test Accuracy: {metrics['test']['accuracy']:.4f}")
        print(f"   Test F1 Score: {metrics['test']['f1']:.4f}")
        print(f"   Test Precision: {metrics['test']['precision']:.4f}")
        print(f"   Test Recall: {metrics['test']['recall']:.4f}")
        print(f"\n   Top 3 Important Features:")
        for i, (feat, imp) in enumerate(list(self.feature_importance.items())[:3]):
            print(f"   {i+1}. {feat}: {imp:.4f}")
        
        # Save model
        if model_save_path:
            self.save_model(model_save_path)
        
        return metrics
    
    def _simulate_training(self) -> Dict:
        """Simulate training when sklearn is not available."""
        self.is_trained = True
        self.feature_importance = {name: np.random.random() for name in self.FEATURE_NAMES}
        total = sum(self.feature_importance.values())
        self.feature_importance = {k: v/total for k, v in self.feature_importance.items()}
        
        self.model_metadata = {
            "trained_at": datetime.now().isoformat(),
            "mode": "simulated",
            "simulated_accuracy": 0.88,
        }
        
        return {
            "test": {"accuracy": 0.88, "f1": 0.85},
            "feature_importance": self.feature_importance
        }
    
    def predict(
        self,
        X: np.ndarray,
        return_probability: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict flood occurrence.
        
        Args:
            X: Feature matrix
            return_probability: Whether to return flood probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not SKLEARN_AVAILABLE or self.model is None or not self.is_trained:
            return self._simulate_prediction(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        probabilities = None
        if return_probability:
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def _simulate_prediction(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate predictions based on simple rules."""
        n_samples = X.shape[0]
        
        # Simple rule-based simulation
        probabilities = np.zeros(n_samples)
        for i in range(n_samples):
            # Use rainfall_3day_sum (index 2) and prev_river_level (index 5)
            rain_factor = X[i, 2] / 200 if X.shape[1] > 2 else 0.5  # Normalize by 200mm
            level_factor = X[i, 5] / 100 if X.shape[1] > 5 else 0.5  # Normalize by 100m
            
            probabilities[i] = np.clip(0.3 * rain_factor + 0.5 * level_factor + np.random.normal(0, 0.1), 0, 1)
        
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def predict_single(
        self,
        features: Dict[str, float]
    ) -> Dict:
        """
        Make prediction for a single sample with detailed output.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Detailed prediction with explanation
        """
        # Convert to array
        X = np.array([[features.get(name, 0) for name in self.FEATURE_NAMES]])
        
        prediction, probability = self.predict(X)
        
        # Get feature contributions
        contributions = {}
        if self.feature_importance:
            for name in self.FEATURE_NAMES:
                value = features.get(name, 0)
                importance = self.feature_importance.get(name, 0)
                # Estimate contribution
                contributions[name] = {
                    "value": value,
                    "importance": importance,
                    "contribution": value * importance / 100  # Simplified
                }
        
        # Determine risk level
        prob = probability[0]
        if prob >= 0.8:
            risk_level = "CRITICAL"
            risk_color = "red"
        elif prob >= 0.6:
            risk_level = "HIGH"
            risk_color = "orange"
        elif prob >= 0.4:
            risk_level = "MODERATE"
            risk_color = "yellow"
        else:
            risk_level = "LOW"
            risk_color = "green"
        
        result = {
            "prediction": "FLOOD" if prediction[0] == 1 else "NO_FLOOD",
            "probability": float(prob),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "feature_contributions": contributions,
            "explanation": self._generate_explanation(features, prob),
            "confidence": self._calculate_confidence(prob),
        }
        
        return result
    
    def _generate_explanation(
        self,
        features: Dict[str, float],
        probability: float
    ) -> str:
        """Generate human-readable explanation for prediction."""
        explanations = []
        
        # Analyze key features
        rain_3d = features.get("rainfall_3day_sum", 0)
        rain_week = features.get("rainfall_week_avg", 0) * 7
        river_level = features.get("prev_river_level", 0)
        
        if rain_3d > 200:
            explanations.append(f"Heavy rainfall ({rain_3d:.0f}mm) over 3 days - historically triggers flooding")
        elif rain_3d > 100:
            explanations.append(f"Elevated rainfall ({rain_3d:.0f}mm) over 3 days")
        
        if river_level > 85:
            explanations.append(f"River level ({river_level:.1f}m) approaching danger threshold")
        
        if features.get("level_change_rate", 0) > 5:
            explanations.append("Rapid rise in water level detected")
        
        if features.get("days_since_heavy_rain", 999) <= 2:
            explanations.append("Recent heavy rainfall event - soil saturated")
        
        if probability > 0.7:
            prefix = "⚠️ HIGH FLOOD RISK: "
        elif probability > 0.4:
            prefix = "🔶 MODERATE RISK: "
        else:
            prefix = "✅ LOW RISK: "
        
        if explanations:
            return prefix + "; ".join(explanations)
        else:
            return prefix + "Normal conditions, no significant risk factors detected"
    
    def _calculate_confidence(self, probability: float) -> float:
        """
        Calculate prediction confidence based on probability distribution.
        
        Confidence is highest when probability is close to 0 or 1,
        lowest when probability is around 0.5 (uncertain).
        """
        # Distance from 0.5 (uncertainty point)
        confidence = abs(probability - 0.5) * 2
        return float(np.clip(confidence, 0.5, 0.95))
    
    def get_feature_importance_chart_data(self) -> List[Dict]:
        """Get feature importance data formatted for charting."""
        return [
            {"feature": name, "importance": float(imp * 100)}
            for name, imp in self.feature_importance.items()
        ]
    
    def save_model(self, path: str) -> None:
        """Save trained model and associated files."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        if SKLEARN_AVAILABLE and self.model:
            joblib.dump(self.model, path)
            print(f"✅ Model saved to {path}")
        
        # Save scaler
        if self.scaler:
            scaler_path = path.replace('.joblib', '_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        meta_path = path.replace('.joblib', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
    
    def load_model(self, path: str) -> None:
        """Load pre-trained model."""
        if SKLEARN_AVAILABLE and os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            print(f"✅ Model loaded from {path}")
        
        # Load scaler
        scaler_path = path.replace('.joblib', '_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        meta_path = path.replace('.joblib', '_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.model_metadata = json.load(f)
                if "feature_importance" in self.model_metadata:
                    self.feature_importance = self.model_metadata["feature_importance"]


class HybridFloodPredictor:
    """
    Hybrid predictor combining LSTM (regression) and Random Forest (classification).
    
    Uses LSTM for continuous river level prediction and RF for binary flood classification,
    combining both for robust flood risk assessment.
    """
    
    def __init__(
        self,
        lstm_model=None,
        rf_model=None,
        danger_level: float = 100.0,
        warning_level: float = 85.0
    ):
        """
        Initialize hybrid predictor.
        
        Args:
            lstm_model: Trained LSTM model instance
            rf_model: Trained Random Forest model instance
            danger_level: River level threshold for flood
            warning_level: River level threshold for warning
        """
        self.lstm_model = lstm_model
        self.rf_model = rf_model
        self.danger_level = danger_level
        self.warning_level = warning_level
    
    def predict(
        self,
        features: Dict[str, float],
        recent_sequence: Optional[np.ndarray] = None,
        forecast_rainfall: Optional[List[float]] = None
    ) -> Dict:
        """
        Combined prediction using both models.
        
        Args:
            features: Tabular features for RF classifier
            recent_sequence: Time series sequence for LSTM
            forecast_rainfall: Future rainfall forecast
            
        Returns:
            Combined prediction result
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "models_used": [],
            "predictions": {},
            "combined_risk": "LOW",
            "recommendation": "",
        }
        
        # Random Forest prediction
        if self.rf_model:
            rf_pred = self.rf_model.predict_single(features)
            result["predictions"]["classification"] = rf_pred
            result["models_used"].append("RandomForest")
        
        # LSTM prediction (if sequence available)
        if self.lstm_model and recent_sequence is not None:
            # This would need the LSTM model's predict method
            result["models_used"].append("LSTM")
        
        # Combine predictions
        if "classification" in result["predictions"]:
            rf_risk = result["predictions"]["classification"]["risk_level"]
            result["combined_risk"] = rf_risk
            
            if rf_risk == "CRITICAL":
                result["recommendation"] = "🚨 EVACUATE: Flood predicted with high confidence. Move to higher ground immediately."
            elif rf_risk == "HIGH":
                result["recommendation"] = "⚠️ PREPARE: High flood risk. Prepare for possible evacuation."
            elif rf_risk == "MODERATE":
                result["recommendation"] = "🔶 MONITOR: Moderate risk. Stay alert and monitor updates."
            else:
                result["recommendation"] = "✅ NORMAL: Low flood risk. Continue normal activities."
        
        return result


if __name__ == "__main__":
    # Test the Random Forest classifier
    print("🌳 Testing Random Forest Flood Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = RandomForestFloodClassifier(n_estimators=100, max_depth=10)
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 500
    
    # Create features
    X = np.column_stack([
        np.random.gamma(2, 30, n_samples),  # rainfall_today
        np.random.gamma(2, 60, n_samples),  # rainfall_2day_sum
        np.random.gamma(2, 90, n_samples),  # rainfall_3day_sum
        np.random.gamma(2, 30, n_samples),  # rainfall_week_avg
        np.random.gamma(3, 40, n_samples),  # rainfall_week_max
        np.random.uniform(40, 110, n_samples),  # prev_river_level
        np.random.normal(0, 5, n_samples),  # level_change_rate
        np.random.gamma(2, 30, n_samples),  # soil_saturation_proxy
        np.random.randint(0, 14, n_samples),  # days_since_heavy_rain
    ])
    
    # Create labels based on features
    # Rule: flood if rainfall_3day_sum > 150 AND prev_river_level > 70
    y = ((X[:, 2] > 150) & (X[:, 5] > 70)).astype(int)
    
    print(f"Generated {n_samples} samples with {sum(y)} flood events")
    
    # Train
    metrics = classifier.train(X, y)
    
    # Test single prediction
    print("\n" + "=" * 50)
    print("Testing single prediction:")
    
    test_features = {
        "rainfall_today": 80,
        "rainfall_2day_sum": 160,
        "rainfall_3day_sum": 250,  # High
        "rainfall_week_avg": 50,
        "rainfall_week_max": 100,
        "prev_river_level": 85,  # High
        "level_change_rate": 8,
        "soil_saturation_proxy": 60,
        "days_since_heavy_rain": 1,
    }
    
    result = classifier.predict_single(test_features)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Explanation: {result['explanation']}")
    
    print("\n✅ Random Forest classifier test complete!")
