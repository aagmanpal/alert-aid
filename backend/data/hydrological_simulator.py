"""
Hydrological Simulator for Flood Forecasting
============================================
Generates synthetic river level data using a simplified watershed model.

Model: river_level_today = 0.8 * river_level_yesterday + 0.2 * rainfall_today - 0.1 * evaporation
Flood Threshold: 100 units (configurable per river)

This module implements the hackathon-specified approach for data simulation
when real historical data is not available.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import json
import os

# River configurations for India
INDIA_RIVERS = {
    "cauvery": {
        "name": "Cauvery River",
        "region": "Karnataka/Tamil Nadu",
        "base_level": 45.0,  # meters
        "danger_level": 100.0,  # meters
        "warning_level": 85.0,  # meters
        "gauge_stations": [
            {"name": "KRS Dam", "lat": 12.4244, "lon": 76.5699},
            {"name": "Mysore", "lat": 12.2958, "lon": 76.6394},
            {"name": "Srirangapatna", "lat": 12.4181, "lon": 76.6947},
        ],
        "avg_monsoon_rainfall": 150,  # mm/day during peak monsoon
        "basin_area_km2": 81155,
    },
    "vrishabhavathi": {
        "name": "Vrishabhavathi River",
        "region": "Bangalore, Karnataka",
        "base_level": 25.0,  # meters
        "danger_level": 60.0,  # meters  
        "warning_level": 50.0,  # meters
        "gauge_stations": [
            {"name": "Kengeri", "lat": 12.9081, "lon": 77.4821},
            {"name": "Nayandahalli", "lat": 12.9553, "lon": 77.5163},
            {"name": "Byramangala", "lat": 12.7947, "lon": 77.3944},
        ],
        "avg_monsoon_rainfall": 80,  # mm/day during peak monsoon
        "basin_area_km2": 388,
    },
    "brahmaputra": {
        "name": "Brahmaputra River",
        "region": "Assam",
        "base_level": 60.0,
        "danger_level": 120.0,
        "warning_level": 100.0,
        "gauge_stations": [
            {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362},
            {"name": "Dibrugarh", "lat": 27.4728, "lon": 94.9120},
        ],
        "avg_monsoon_rainfall": 200,
        "basin_area_km2": 580000,
    },
    "yamuna": {
        "name": "Yamuna River",
        "region": "Delhi NCR",
        "base_level": 203.0,  # Normal water level at Old Railway Bridge, Delhi
        "danger_level": 207.0,  # Official danger mark at Delhi
        "warning_level": 205.5,  # Warning level
        "gauge_stations": [
            {"name": "Old Railway Bridge", "lat": 28.6692, "lon": 77.2311},
            {"name": "ITO Barrage", "lat": 28.6194, "lon": 77.2453},
            {"name": "Okhla Barrage", "lat": 28.5458, "lon": 77.3028},
        ],
        "avg_monsoon_rainfall": 180,  # mm/day during peak monsoon
        "basin_area_km2": 366223,
    }
}


class HydrologicalSimulator:
    """
    Simulates river water levels using a simplified hydrological model.
    
    The model captures:
    1. Memory effect: River level depends on previous day (coefficient 0.8)
    2. Rainfall contribution: New water from rain (coefficient 0.2)
    3. Evaporation/drainage: Water loss (coefficient 0.1)
    
    Formula: river_level[t] = 0.8 * river_level[t-1] + 0.2 * rainfall[t] - evaporation
    """
    
    def __init__(
        self,
        river_id: str = "cauvery",
        memory_coefficient: float = 0.8,
        rainfall_coefficient: float = 0.2,
        evaporation_rate: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize the hydrological simulator.
        
        Args:
            river_id: ID of the river (cauvery, vrishabhavathi, brahmaputra)
            memory_coefficient: How much previous day's level affects today (0.8)
            rainfall_coefficient: How much rainfall contributes to level (0.2)
            evaporation_rate: Daily water loss rate (0.1)
            random_seed: For reproducible results
        """
        self.river_id = river_id
        self.river_config = INDIA_RIVERS.get(river_id, INDIA_RIVERS["cauvery"])
        
        self.memory_coef = memory_coefficient
        self.rainfall_coef = rainfall_coefficient
        self.evap_rate = evaporation_rate
        
        self.base_level = self.river_config["base_level"]
        self.danger_level = self.river_config["danger_level"]
        self.warning_level = self.river_config["warning_level"]
        
        np.random.seed(random_seed)
        
    def generate_rainfall_pattern(
        self,
        num_days: int = 365,
        start_date: Optional[datetime] = None,
        include_monsoon: bool = True
    ) -> pd.DataFrame:
        """
        Generate realistic rainfall patterns for Indian monsoon climate.
        
        Monsoon season (June-September): Heavy rainfall
        Post-monsoon (October-November): Moderate rainfall
        Winter (December-February): Low rainfall
        Pre-monsoon (March-May): Scattered rainfall
        
        Args:
            num_days: Number of days to simulate
            start_date: Starting date for simulation
            include_monsoon: Whether to include seasonal patterns
            
        Returns:
            DataFrame with date and rainfall_mm columns
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1)
            
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        rainfall = np.zeros(num_days)
        
        for i, date in enumerate(dates):
            month = date.month
            
            if include_monsoon:
                # Monsoon season (June-September): Heavy rainfall
                if month in [6, 7, 8, 9]:
                    # Base monsoon rainfall with high variability
                    base_rain = self.river_config["avg_monsoon_rainfall"]
                    rainfall[i] = max(0, np.random.gamma(2, base_rain / 2))
                    
                    # Add extreme events (10% chance of very heavy rain)
                    if np.random.random() < 0.10:
                        rainfall[i] *= np.random.uniform(1.5, 3.0)
                        
                # Post-monsoon (October-November)
                elif month in [10, 11]:
                    rainfall[i] = max(0, np.random.gamma(1.5, 30))
                    
                # Winter (December-February): Very low
                elif month in [12, 1, 2]:
                    rainfall[i] = max(0, np.random.exponential(5))
                    
                # Pre-monsoon (March-May): Scattered thunderstorms
                else:
                    if np.random.random() < 0.3:  # 30% chance of rain
                        rainfall[i] = np.random.gamma(2, 20)
                    else:
                        rainfall[i] = 0
            else:
                # Uniform random rainfall (for testing)
                rainfall[i] = max(0, np.random.exponential(50))
        
        return pd.DataFrame({
            "date": dates,
            "rainfall_mm": np.round(rainfall, 2)
        })
    
    def simulate_river_level(
        self,
        rainfall_data: pd.DataFrame,
        initial_level: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Simulate river water levels using the hydrological model.
        
        Model: level[t] = 0.8 * level[t-1] + 0.2 * rainfall[t] - evaporation
        
        Args:
            rainfall_data: DataFrame with date and rainfall_mm columns
            initial_level: Starting river level (defaults to base_level)
            
        Returns:
            DataFrame with river level time series
        """
        if initial_level is None:
            initial_level = self.base_level
            
        n_days = len(rainfall_data)
        river_levels = np.zeros(n_days)
        river_levels[0] = initial_level
        
        rainfall = rainfall_data["rainfall_mm"].values
        
        for t in range(1, n_days):
            # Core hydrological model
            memory_effect = self.memory_coef * river_levels[t-1]
            rainfall_effect = self.rainfall_coef * rainfall[t]
            evaporation = self.evap_rate * self.base_level
            
            # Add small random noise for realism
            noise = np.random.normal(0, 1)
            
            river_levels[t] = max(
                self.base_level * 0.5,  # Minimum level
                memory_effect + rainfall_effect - evaporation + noise
            )
            
        # Create output dataframe
        result = rainfall_data.copy()
        result["river_level_m"] = np.round(river_levels, 2)
        result["is_flood"] = (river_levels >= self.danger_level).astype(int)
        result["is_warning"] = (river_levels >= self.warning_level).astype(int)
        
        # Create bins ensuring monotonically increasing values
        # Handle rivers where base_level * 1.2 might exceed warning/danger levels (e.g., Yamuna)
        low_threshold = min(self.base_level * 1.2, self.warning_level * 0.95)
        bins = sorted([0, low_threshold, self.warning_level, self.danger_level, float('inf')])
        # Remove duplicates while preserving order
        bins = list(dict.fromkeys(bins))
        
        # Adjust labels based on number of bins
        if len(bins) == 5:
            labels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        elif len(bins) == 4:
            labels = ["LOW", "HIGH", "CRITICAL"]
        else:
            labels = ["LOW", "CRITICAL"]
            
        result["risk_level"] = pd.cut(
            river_levels,
            bins=bins,
            labels=labels
        )
        
        return result
    
    def extract_features(
        self,
        data: pd.DataFrame,
        lookback_days: int = 7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features for ML model training.
        
        Features:
        1. rainfall_today: Current day's rainfall
        2. rainfall_2day_sum: Sum of last 2 days rainfall
        3. rainfall_week_avg: Average rainfall over last 7 days
        4. prev_river_level: Previous day's river level
        5. rainfall_3day_sum: Sum of last 3 days rainfall
        6. level_change_rate: Rate of change in river level
        7. soil_saturation_proxy: Cumulative recent rainfall as saturation indicator
        
        Args:
            data: DataFrame from simulate_river_level
            lookback_days: Days of history to use for features
            
        Returns:
            Tuple of (features, regression_targets, classification_targets)
        """
        n_samples = len(data) - lookback_days
        
        # Feature arrays
        features = []
        reg_targets = []  # Next day river level
        cls_targets = []  # Flood / no-flood
        
        rainfall = data["rainfall_mm"].values
        river_level = data["river_level_m"].values
        is_flood = data["is_flood"].values
        
        for i in range(lookback_days, len(data) - 1):
            # Extract features
            feat = {
                "rainfall_today": rainfall[i],
                "rainfall_2day_sum": np.sum(rainfall[i-1:i+1]),
                "rainfall_3day_sum": np.sum(rainfall[i-2:i+1]),
                "rainfall_week_avg": np.mean(rainfall[i-6:i+1]),
                "rainfall_week_max": np.max(rainfall[i-6:i+1]),
                "prev_river_level": river_level[i],
                "level_change_rate": river_level[i] - river_level[i-1],
                "soil_saturation_proxy": np.sum(rainfall[i-6:i+1]) / 7,
                "days_since_heavy_rain": self._days_since_heavy_rain(rainfall[:i+1]),
            }
            features.append(list(feat.values()))
            
            # Targets: predict next day
            reg_targets.append(river_level[i + 1])
            cls_targets.append(is_flood[i + 1])
        
        feature_names = list(feat.keys())
        
        return (
            np.array(features),
            np.array(reg_targets),
            np.array(cls_targets),
            feature_names
        )
    
    def _days_since_heavy_rain(self, rainfall_history: np.ndarray, threshold: float = 50) -> int:
        """Calculate days since last heavy rainfall event."""
        heavy_rain_days = np.where(rainfall_history >= threshold)[0]
        if len(heavy_rain_days) == 0:
            return len(rainfall_history)
        return len(rainfall_history) - heavy_rain_days[-1] - 1
    
    def create_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: DataFrame with rainfall and river level
            sequence_length: Number of time steps in each sequence
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        features = np.column_stack([
            data["rainfall_mm"].values,
            data["river_level_m"].values
        ])
        
        X, y = [], []
        
        for i in range(sequence_length, len(features) - 1):
            X.append(features[i-sequence_length:i])
            y.append(features[i + 1, 1])  # Next day river level
            
        return np.array(X), np.array(y)
    
    def generate_full_dataset(
        self,
        num_days: int = 730,  # 2 years
        train_ratio: float = 0.8,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate complete training dataset with all features.
        
        Args:
            num_days: Total days to simulate
            train_ratio: Fraction for training (rest for testing)
            save_path: Optional path to save CSV files
            
        Returns:
            Dictionary with train/test splits and metadata
        """
        # Generate data
        rainfall_df = self.generate_rainfall_pattern(num_days)
        full_data = self.simulate_river_level(rainfall_df)
        
        # Extract features
        X, y_reg, y_cls, feature_names = self.extract_features(full_data)
        
        # Train/test split
        split_idx = int(len(X) * train_ratio)
        
        dataset = {
            "X_train": X[:split_idx],
            "X_test": X[split_idx:],
            "y_train_regression": y_reg[:split_idx],
            "y_test_regression": y_reg[split_idx:],
            "y_train_classification": y_cls[:split_idx],
            "y_test_classification": y_cls[split_idx:],
            "feature_names": feature_names,
            "full_timeseries": full_data,
            "metadata": {
                "river_id": self.river_id,
                "river_name": self.river_config["name"],
                "danger_level": self.danger_level,
                "warning_level": self.warning_level,
                "num_flood_days": int(full_data["is_flood"].sum()),
                "flood_percentage": float(full_data["is_flood"].mean() * 100),
                "generated_at": datetime.now().isoformat(),
            }
        }
        
        # Save to CSV if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            # Save full timeseries
            full_data.to_csv(
                os.path.join(save_path, f"{self.river_id}_timeseries.csv"),
                index=False
            )
            
            # Save training features
            train_df = pd.DataFrame(X[:split_idx], columns=feature_names)
            train_df["target_level"] = y_reg[:split_idx]
            train_df["target_flood"] = y_cls[:split_idx]
            train_df.to_csv(
                os.path.join(save_path, f"{self.river_id}_train.csv"),
                index=False
            )
            
            # Save test features
            test_df = pd.DataFrame(X[split_idx:], columns=feature_names)
            test_df["target_level"] = y_reg[split_idx:]
            test_df["target_flood"] = y_cls[split_idx:]
            test_df.to_csv(
                os.path.join(save_path, f"{self.river_id}_test.csv"),
                index=False
            )
            
            # Save metadata
            with open(os.path.join(save_path, f"{self.river_id}_metadata.json"), "w") as f:
                json.dump(dataset["metadata"], f, indent=2)
                
            print(f"✅ Saved dataset to {save_path}")
        
        return dataset
    
    def get_prediction_context(
        self,
        current_rainfall: float,
        recent_rainfall: List[float],
        current_level: float
    ) -> Dict:
        """
        Create prediction context for real-time forecasting.
        
        Args:
            current_rainfall: Today's rainfall in mm
            recent_rainfall: Last 7 days rainfall [day-6, day-5, ..., day-1]
            current_level: Current river level in meters
            
        Returns:
            Feature dictionary for model prediction
        """
        all_rainfall = recent_rainfall + [current_rainfall]
        
        return {
            "rainfall_today": current_rainfall,
            "rainfall_2day_sum": sum(all_rainfall[-2:]),
            "rainfall_3day_sum": sum(all_rainfall[-3:]),
            "rainfall_week_avg": np.mean(all_rainfall),
            "rainfall_week_max": max(all_rainfall),
            "prev_river_level": current_level,
            "level_change_rate": 0,  # Would need previous level
            "soil_saturation_proxy": sum(all_rainfall) / 7,
            "days_since_heavy_rain": self._days_since_heavy_rain(np.array(all_rainfall)),
        }


def generate_all_river_datasets(output_dir: str = "backend/data"):
    """Generate datasets for all configured Indian rivers."""
    
    print("🌊 Generating Flood Forecasting Datasets for Indian Rivers")
    print("=" * 60)
    
    for river_id in INDIA_RIVERS.keys():
        print(f"\n📍 Processing: {INDIA_RIVERS[river_id]['name']}")
        
        simulator = HydrologicalSimulator(river_id=river_id)
        dataset = simulator.generate_full_dataset(
            num_days=730,  # 2 years
            save_path=output_dir
        )
        
        meta = dataset["metadata"]
        print(f"   ├── Generated {len(dataset['X_train'])} training samples")
        print(f"   ├── Generated {len(dataset['X_test'])} test samples")
        print(f"   ├── Flood days: {meta['num_flood_days']} ({meta['flood_percentage']:.1f}%)")
        print(f"   └── Danger level: {meta['danger_level']}m")
    
    print("\n✅ All datasets generated successfully!")


if __name__ == "__main__":
    # Generate all datasets when run directly
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "backend/data"
    generate_all_river_datasets(output_dir)
