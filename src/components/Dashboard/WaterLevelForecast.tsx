import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
  Bar
} from 'recharts';
import { AlertTriangle, Droplets, TrendingUp, Activity, RefreshCw } from 'lucide-react';

// Types
interface TimeSeriesPoint {
  date: string;
  rainfall_mm: number;
  river_level_m: number;
  is_flood: boolean;
  risk_level: string;
}

interface Prediction {
  predicted_level_m: number;
  forecast_rainfall_mm: number;
  risk_level: string;
  above_danger: boolean;
  above_warning: boolean;
}

interface RiverInfo {
  id: string;
  name: string;
  region: string;
  danger_level: number;
  warning_level: number;
  base_level: number;
}

interface WaterLevelForecastProps {
  riverId?: string;
  currentRainfall?: number;
  currentLevel?: number;
  forecastRainfall?: number[];
  onPredictionUpdate?: (data: any) => void;
}

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Custom tooltip component
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl">
        <p className="text-white font-semibold mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }} className="text-sm">
            {entry.name}: {entry.value?.toFixed(1)} {entry.name.includes('Level') ? 'm' : 'mm'}
          </p>
        ))}
        {data.risk_level && (
          <p className={`text-sm mt-2 font-medium ${
            data.risk_level === 'CRITICAL' ? 'text-red-400' :
            data.risk_level === 'HIGH' ? 'text-orange-400' :
            data.risk_level === 'MODERATE' ? 'text-yellow-400' :
            'text-green-400'
          }`}>
            Risk: {data.risk_level}
          </p>
        )}
      </div>
    );
  }
  return null;
};

// Risk badge component
const RiskBadge: React.FC<{ level: string }> = ({ level }) => {
  const colors: Record<string, string> = {
    CRITICAL: 'bg-red-500 text-white animate-pulse',
    HIGH: 'bg-orange-500 text-white',
    MODERATE: 'bg-yellow-500 text-black',
    LOW: 'bg-green-500 text-white',
  };

  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${colors[level] || colors.LOW}`}>
      {level}
    </span>
  );
};

export const WaterLevelForecast: React.FC<WaterLevelForecastProps> = ({
  riverId = 'cauvery',
  currentRainfall = 50,
  currentLevel = 60,
  forecastRainfall = [60, 70, 50],
  onPredictionUpdate
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [predictionData, setPredictionData] = useState<any>(null);
  const [selectedRiver, setSelectedRiver] = useState(riverId);
  const [rivers, setRivers] = useState<RiverInfo[]>([]);

  // Fetch available rivers
  useEffect(() => {
    const fetchRivers = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/flood/india/rivers`);
        if (response.ok) {
          const data = await response.json();
          setRivers(data);
        }
      } catch (err) {
        console.error('Failed to fetch rivers:', err);
        // Use default rivers
        setRivers([
          { id: 'cauvery', name: 'Cauvery River', region: 'Karnataka/Tamil Nadu', danger_level: 100, warning_level: 85, base_level: 45 },
          { id: 'vrishabhavathi', name: 'Vrishabhavathi River', region: 'Bangalore', danger_level: 60, warning_level: 50, base_level: 25 },
          { id: 'brahmaputra', name: 'Brahmaputra River', region: 'Assam', danger_level: 120, warning_level: 100, base_level: 60 },
        ]);
      }
    };
    fetchRivers();
  }, []);

  // Fetch prediction data
  useEffect(() => {
    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`${API_BASE_URL}/api/flood/india/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            river_id: selectedRiver,
            current_rainfall: currentRainfall,
            recent_rainfall: [30, 40, 50, 60, 70, 45, currentRainfall * 0.8],
            current_river_level: currentLevel,
            forecast_rainfall: forecastRainfall,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        setPredictionData(data);
        onPredictionUpdate?.(data);
      } catch (err) {
        console.error('Prediction error:', err);
        setError('Failed to fetch prediction. Using simulated data.');
        // Generate simulated data
        setPredictionData(generateSimulatedData(selectedRiver, currentRainfall, currentLevel, forecastRainfall));
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, [selectedRiver, currentRainfall, currentLevel, forecastRainfall, onPredictionUpdate]);

  // Prepare chart data
  const chartData = useMemo(() => {
    if (!predictionData) return [];

    const data: any[] = [];
    
    // Historical data
    predictionData.time_series?.forEach((point: TimeSeriesPoint) => {
      data.push({
        date: point.date.slice(5), // MM-DD format
        actual_level: point.river_level_m,
        rainfall: point.rainfall_mm,
        risk_level: point.risk_level,
        type: 'historical'
      });
    });

    // Prediction data
    const today = new Date();
    Object.entries(predictionData.predictions || {}).forEach(([horizon, pred]: [string, any], index) => {
      const futureDate = new Date(today);
      futureDate.setDate(futureDate.getDate() + index + 1);
      const dateStr = `${String(futureDate.getMonth() + 1).padStart(2, '0')}-${String(futureDate.getDate()).padStart(2, '0')}`;
      
      data.push({
        date: dateStr,
        predicted_level: pred.predicted_level_m,
        forecast_rainfall: pred.forecast_rainfall_mm,
        risk_level: pred.risk_level,
        type: 'forecast'
      });
    });

    return data;
  }, [predictionData]);

  const river = rivers.find(r => r.id === selectedRiver) || rivers[0];
  const dangerLevel = river?.danger_level || 100;
  const warningLevel = river?.warning_level || 85;

  if (loading && !predictionData) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 animate-pulse">
        <div className="h-8 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-64 bg-gray-700 rounded"></div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-xl p-6 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Droplets className="w-6 h-6 text-blue-400" />
          <h2 className="text-xl font-bold text-white">Water Level Forecast</h2>
        </div>
        
        {/* River selector */}
        <select
          value={selectedRiver}
          onChange={(e) => setSelectedRiver(e.target.value)}
          className="bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {rivers.map(r => (
            <option key={r.id} value={r.id}>{r.name}</option>
          ))}
        </select>
      </div>

      {/* Error message */}
      {error && (
        <div className="bg-yellow-500/20 border border-yellow-500 rounded-lg p-3 mb-4 text-yellow-200 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Current conditions cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-700/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Current Level</p>
          <p className="text-2xl font-bold text-white">{currentLevel.toFixed(1)}m</p>
          <p className="text-xs text-gray-400">{((currentLevel / dangerLevel) * 100).toFixed(0)}% of danger</p>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Today's Rainfall</p>
          <p className="text-2xl font-bold text-blue-400">{currentRainfall}mm</p>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm">1-Day Forecast</p>
          <p className="text-2xl font-bold text-orange-400">
            {predictionData?.predictions?.['1d']?.predicted_level_m?.toFixed(1) || '--'}m
          </p>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Risk Level</p>
          <div className="mt-1">
            <RiskBadge level={predictionData?.risk_assessment?.risk_level || 'LOW'} />
          </div>
        </div>
      </div>

      {/* Main chart */}
      <div className="h-80 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="date" 
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 12 }}
            />
            <YAxis 
              yAxisId="level"
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 12 }}
              label={{ value: 'Water Level (m)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
            />
            <YAxis 
              yAxisId="rainfall"
              orientation="right"
              stroke="#60A5FA"
              tick={{ fill: '#60A5FA', fontSize: 12 }}
              label={{ value: 'Rainfall (mm)', angle: 90, position: 'insideRight', fill: '#60A5FA' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {/* Danger level line */}
            <ReferenceLine 
              y={dangerLevel} 
              yAxisId="level"
              stroke="#EF4444" 
              strokeDasharray="5 5" 
              strokeWidth={2}
              label={{ value: 'Danger Level', fill: '#EF4444', fontSize: 12 }}
            />
            
            {/* Warning level line */}
            <ReferenceLine 
              y={warningLevel} 
              yAxisId="level"
              stroke="#F59E0B" 
              strokeDasharray="3 3" 
              strokeWidth={2}
              label={{ value: 'Warning Level', fill: '#F59E0B', fontSize: 12 }}
            />

            {/* Rainfall bars */}
            <Bar 
              dataKey="rainfall" 
              yAxisId="rainfall"
              fill="#3B82F6" 
              opacity={0.6}
              name="Actual Rainfall"
            />
            <Bar 
              dataKey="forecast_rainfall" 
              yAxisId="rainfall"
              fill="#60A5FA" 
              opacity={0.4}
              name="Forecast Rainfall"
            />

            {/* Actual water level line */}
            <Line
              type="monotone"
              dataKey="actual_level"
              yAxisId="level"
              stroke="#10B981"
              strokeWidth={2}
              dot={{ fill: '#10B981', strokeWidth: 2 }}
              name="Actual Level"
              connectNulls
            />

            {/* Predicted water level line */}
            <Line
              type="monotone"
              dataKey="predicted_level"
              yAxisId="level"
              stroke="#F59E0B"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#F59E0B', strokeWidth: 2 }}
              name="Predicted Level"
              connectNulls
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Alerts section */}
      {predictionData?.alerts && predictionData.alerts.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            Alerts
          </h3>
          <div className="space-y-2">
            {predictionData.alerts.map((alert: any, index: number) => (
              <div
                key={index}
                className={`rounded-lg p-3 ${
                  alert.severity === 'CRITICAL' ? 'bg-red-500/20 border border-red-500' :
                  alert.severity === 'HIGH' ? 'bg-orange-500/20 border border-orange-500' :
                  alert.severity === 'MODERATE' ? 'bg-yellow-500/20 border border-yellow-500' :
                  'bg-green-500/20 border border-green-500'
                }`}
              >
                <p className="text-white">{alert.message}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk explanation */}
      {predictionData?.risk_assessment?.explanation && (
        <div className="bg-gray-700/50 rounded-lg p-4 mb-6">
          <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            AI Analysis
          </h3>
          <p className="text-gray-300">{predictionData.risk_assessment.explanation}</p>
          <div className="mt-3 flex items-center gap-4">
            <span className="text-sm text-gray-400">
              Flood Probability: <span className="text-white font-medium">
                {((predictionData.risk_assessment.flood_probability || 0) * 100).toFixed(0)}%
              </span>
            </span>
            <span className="text-sm text-gray-400">
              Confidence: <span className="text-white font-medium">
                {((predictionData.risk_assessment.confidence || 0) * 100).toFixed(0)}%
              </span>
            </span>
          </div>
        </div>
      )}

      {/* Feature importance */}
      {predictionData?.feature_importance && predictionData.feature_importance.length > 0 && (
        <div className="bg-gray-700/50 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            Key Risk Factors
          </h3>
          <div className="space-y-2">
            {predictionData.feature_importance.slice(0, 5).map((feat: any, index: number) => (
              <div key={index} className="flex items-center gap-3">
                <span className="text-gray-400 text-sm w-40 truncate">
                  {feat.feature.replace(/_/g, ' ')}
                </span>
                <div className="flex-1 bg-gray-600 rounded-full h-2">
                  <div
                    className="bg-cyan-400 h-2 rounded-full"
                    style={{ width: `${Math.min(feat.importance, 100)}%` }}
                  />
                </div>
                <span className="text-white text-sm w-12">
                  {feat.importance.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model info footer */}
      <div className="mt-4 pt-4 border-t border-gray-700 flex items-center justify-between text-xs text-gray-500">
        <span>Model: {predictionData?.model_info?.classification_model || 'Random Forest'} + {predictionData?.model_info?.regression_model || 'LSTM'}</span>
        <span>Accuracy: {((predictionData?.model_info?.accuracy || 0.88) * 100).toFixed(1)}%</span>
        <button
          onClick={() => window.location.reload()}
          className="flex items-center gap-1 text-gray-400 hover:text-white transition-colors"
        >
          <RefreshCw className="w-3 h-3" />
          Refresh
        </button>
      </div>
    </div>
  );
};

// Helper function to generate simulated data when API is unavailable
function generateSimulatedData(
  riverId: string,
  currentRainfall: number,
  currentLevel: number,
  forecastRainfall: number[]
): any {
  const riverConfigs: Record<string, { danger_level: number; warning_level: number; base_level: number }> = {
    cauvery: { danger_level: 100, warning_level: 85, base_level: 45 },
    vrishabhavathi: { danger_level: 60, warning_level: 50, base_level: 25 },
    brahmaputra: { danger_level: 120, warning_level: 100, base_level: 60 },
  };

  const config = riverConfigs[riverId] || riverConfigs.cauvery;
  
  // Generate historical time series
  const today = new Date();
  const timeSeries: TimeSeriesPoint[] = [];
  let level = config.base_level;

  for (let i = 7; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const rainfall = 30 + Math.random() * 50;
    level = 0.8 * level + 0.2 * rainfall - 0.1 * config.base_level + (Math.random() - 0.5) * 5;
    level = Math.max(config.base_level * 0.5, level);

    timeSeries.push({
      date: date.toISOString().split('T')[0],
      rainfall_mm: Math.round(rainfall * 10) / 10,
      river_level_m: Math.round(level * 10) / 10,
      is_flood: level >= config.danger_level,
      risk_level: level >= config.danger_level ? 'CRITICAL' :
                  level >= config.warning_level ? 'HIGH' :
                  level >= config.warning_level * 0.8 ? 'MODERATE' : 'LOW'
    });
  }

  // Generate predictions
  const predictions: Record<string, Prediction> = {};
  let predLevel = currentLevel;

  forecastRainfall.forEach((rain, index) => {
    predLevel = 0.8 * predLevel + 0.2 * rain - 0.1 * config.base_level;
    const horizon = `${index + 1}d`;
    predictions[horizon] = {
      predicted_level_m: Math.round(predLevel * 10) / 10,
      forecast_rainfall_mm: rain,
      risk_level: predLevel >= config.danger_level ? 'CRITICAL' :
                  predLevel >= config.warning_level ? 'HIGH' :
                  predLevel >= config.warning_level * 0.8 ? 'MODERATE' : 'LOW',
      above_danger: predLevel >= config.danger_level,
      above_warning: predLevel >= config.warning_level,
    };
  });

  // Calculate flood probability based on conditions
  const floodProbability = Math.min(0.95, 
    (currentLevel / config.danger_level) * 0.5 + 
    (currentRainfall / 100) * 0.3 +
    (forecastRainfall.reduce((a, b) => a + b, 0) / 300) * 0.2
  );

  return {
    river: {
      id: riverId,
      name: riverId.charAt(0).toUpperCase() + riverId.slice(1) + ' River',
      danger_level: config.danger_level,
      warning_level: config.warning_level,
    },
    current_conditions: {
      rainfall_mm: currentRainfall,
      river_level_m: currentLevel,
    },
    predictions,
    risk_assessment: {
      classification: floodProbability > 0.5 ? 'FLOOD' : 'NO_FLOOD',
      flood_probability: floodProbability,
      risk_level: floodProbability > 0.8 ? 'CRITICAL' :
                  floodProbability > 0.6 ? 'HIGH' :
                  floodProbability > 0.4 ? 'MODERATE' : 'LOW',
      confidence: 0.75 + Math.random() * 0.15,
      explanation: floodProbability > 0.6 
        ? `⚠️ Elevated flood risk detected. Heavy rainfall (${currentRainfall}mm) combined with rising river levels. Monitor conditions closely.`
        : `✅ Normal conditions. Current rainfall and river levels within safe parameters.`
    },
    alerts: floodProbability > 0.6 ? [{
      type: 'FLOOD_WATCH',
      severity: floodProbability > 0.8 ? 'CRITICAL' : 'HIGH',
      message: `🔶 ${floodProbability > 0.8 ? 'High flood probability' : 'Elevated flood risk'} detected. Monitor river levels and prepare for possible action.`
    }] : [{
      type: 'ALL_CLEAR',
      severity: 'INFO',
      message: '✅ No significant flood risk detected. Normal conditions expected.'
    }],
    feature_importance: [
      { feature: 'prev_river_level', importance: 35 },
      { feature: 'rainfall_3day_sum', importance: 25 },
      { feature: 'rainfall_today', importance: 15 },
      { feature: 'soil_saturation_proxy', importance: 12 },
      { feature: 'level_change_rate', importance: 8 },
      { feature: 'days_since_heavy_rain', importance: 5 },
    ],
    time_series: timeSeries,
    model_info: {
      classification_model: 'Random Forest (Simulated)',
      regression_model: 'Hydrological Model',
      accuracy: 0.88,
    }
  };
}

export default WaterLevelForecast;
