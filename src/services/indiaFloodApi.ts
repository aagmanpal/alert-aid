/**
 * India Flood Forecasting API Service
 * Connects to India-specific flood prediction endpoints
 * Implements hydrological model: river_level = 0.8*prev + 0.2*rainfall - 0.1*evaporation
 * 
 * For Hackathon: "Flood Forecasting and Disaster Warning – AI for Disaster Management"
 */

const getBackendUrl = (): string => {
  if (process.env.REACT_APP_BACKEND_URL) {
    return process.env.REACT_APP_BACKEND_URL;
  }
  return 'http://localhost:8000';
};

const API_BASE_URL = getBackendUrl();

// Cache configuration for India flood data
const CACHE_DURATION = 3 * 60 * 1000; // 3 minutes for real-time flood data
const cache: Map<string, { data: any; timestamp: number }> = new Map();

function getCacheKey(endpoint: string, params?: Record<string, any>): string {
  return params ? `${endpoint}:${JSON.stringify(params)}` : endpoint;
}

function getFromCache<T>(key: string): T | null {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    console.log(`📦 India Flood Cache hit: ${key.split(':')[0]}`);
    return cached.data as T;
  }
  return null;
}

function setCache(key: string, data: any): void {
  cache.set(key, { data, timestamp: Date.now() });
  console.log(`💾 India Flood Cached: ${key.split(':')[0]}`);
}

export function clearIndiaFloodCache(): void {
  cache.clear();
  console.log('🗑️ India Flood API cache cleared');
}

// =============================================================================
// Types for India Flood Forecasting
// =============================================================================

export interface IndiaRiver {
  id: string;
  name: string;
  region?: string;  // API uses 'region' instead of 'state'
  basin?: string;
  state?: string;
  base_level: number;
  danger_level: number;
  warning_level: number;
  flood_level?: number;
  catchment_area_km2?: number;
  basin_area_km2?: number;  // API uses this
  avg_annual_rainfall_mm?: number;
  avg_monsoon_rainfall?: number;  // API uses this
  monsoon_peak_months?: string[];
  gauge_stations?: Array<{name: string; lat: number; lon: number}>;
}

export interface WaterLevelPrediction {
  hour: number;
  predicted_level: number;
  confidence_lower: number;
  confidence_upper: number;
  flood_probability: number;
}

export interface FloodPredictionResult {
  river_id: string;
  river_name: string;
  current_level: number;
  danger_level: number;
  warning_level: number;
  predictions: WaterLevelPrediction[];
  risk_assessment: {
    current_risk: 'low' | 'moderate' | 'high' | 'critical';
    trend: 'rising' | 'stable' | 'falling';
    hours_to_danger: number | null;
    max_predicted_level: number;
    flood_probability_24h: number;
  };
  alerts: {
    level: 'info' | 'warning' | 'danger' | 'critical';
    message: string;
    recommended_action: string;
  }[];
  feature_importance: {
    feature: string;
    importance: number;
  }[];
  model_info: {
    rf_accuracy: number;
    lstm_confidence: number;
    last_trained: string;
  };
  ai_analysis: string;
  timestamp: string;
}

export interface SimulationRequest {
  river_id: string;
  current_level: number;
  rainfall_today: number;
  forecast_rainfall: number[];  // Next 7 days
  soil_saturation?: number;  // Percentage (0-100)
  upstream_release?: number;  // m³/s
}

export interface SimulationResult {
  simulation_id: string;
  river_id: string;
  river_name: string;
  input_parameters: {
    current_level: number;
    rainfall_today: number;
    forecast_rainfall: number[];
    soil_saturation?: number;
    upstream_release?: number;
  };
  thresholds: {
    danger_level: number;
    warning_level: number;
    base_level: number;
  };
  predictions: {
    hour: number;
    predicted_level: number;
    confidence_lower: number;
    confidence_upper: number;
    flood_probability: number;
    risk_level: string;
  }[];
  risk_assessment: {
    overall_risk: string;
    hours_to_danger: number | null;
    max_level: number;
    min_level: number;
    avg_level: number;
    flood_probability_24h: number;
  };
  model_info: {
    model_type: string;
    formula: string;
    parameters: {
      memory_coefficient: number;
      rainfall_coefficient: number;
      evaporation_rate: number;
    };
  };
  computation_time_ms: number;
  timestamp: string;
}

export interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  latency_ms: number;
  components: {
    api: string;
    random_forest: string;
    lstm: string;
    hydrological_model: string;
    rivers_database: string;
  };
  rivers_supported: string[];
  version: string;
}

export interface ModelStatus {
  models_trained: boolean;
  ensemble_mode: boolean;
  random_forest: {
    status: string;
    accuracy: number;
    f1_score: number;
    feature_count: number;
    n_estimators: number;
    last_trained: string;
  };
  lstm: {
    status: string;
    framework: string;
    architecture: string;
    mae?: number;
    rmse?: number;
    r2_score?: number;
    parameters?: number;
    hidden_size?: number;
    num_layers?: number;
    sequence_length?: number;
    output_horizons?: number;
    last_trained?: string;
  };
  rivers_supported: string[];
  hydrological_model: {
    formula: string;
    memory_coefficient: number;
    rainfall_coefficient: number;
    evaporation_rate: number;
  };
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Get list of supported Indian rivers with their characteristics
 * Uses v2 frontend-optimized API
 */
export async function getIndiaRivers(): Promise<IndiaRiver[]> {
  const cacheKey = getCacheKey('india-rivers');
  const cached = getFromCache<IndiaRiver[]>(cacheKey);
  if (cached) return cached;

  try {
    // Use v2 frontend-optimized API endpoint
    const response = await fetch(`${API_BASE_URL}/api/flood/india/v2/rivers`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    // API returns array directly, not wrapped in {rivers: [...]}
    const rivers = Array.isArray(data) ? data : (data.rivers || []);
    setCache(cacheKey, rivers);
    return rivers;
  } catch (error) {
    console.error('Failed to fetch India rivers:', error);
    throw error;
  }
}

/**
 * Get flood prediction for a specific river
 * Uses the v2 frontend-optimized API that returns data in the exact format needed
 * @param useEnsemble - If true, uses advanced endpoint with LSTM ensemble
 */
export async function getRiverPrediction(
  riverId: string,
  skipCache: boolean = false,
  useEnsemble: boolean = true
): Promise<FloodPredictionResult> {
  const cacheKey = getCacheKey('river-prediction', { riverId, useEnsemble });
  
  if (!skipCache) {
    const cached = getFromCache<FloodPredictionResult>(cacheKey);
    if (cached) return cached;
  }

  try {
    // Use advanced endpoint with LSTM ensemble if available
    const endpoint = useEnsemble 
      ? `${API_BASE_URL}/api/flood/india/v2/predict-advanced/${riverId}`
      : `${API_BASE_URL}/api/flood/india/v2/predict/${riverId}`;
    
    const response = await fetch(endpoint);
    if (!response.ok) {
      // Fallback to basic endpoint if advanced fails
      if (useEnsemble) {
        console.log('Advanced endpoint failed, falling back to basic');
        return getRiverPrediction(riverId, skipCache, false);
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    setCache(cacheKey, data);
    return data;
  } catch (error) {
    console.error(`Failed to fetch prediction for river ${riverId}:`, error);
    throw error;
  }
}

/**
 * Get flood prediction with custom parameters
 */
export async function getCustomPrediction(
  riverId: string,
  currentLevel: number,
  rainfallToday: number,
  forecastRainfall: number[]
): Promise<FloodPredictionResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/flood/india/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        river_id: riverId,
        current_level: currentLevel,
        rainfall_today: rainfallToday,
        forecast_rainfall: forecastRainfall,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to get custom prediction:', error);
    throw error;
  }
}

/**
 * Run what-if simulation for flood scenarios
 * Uses the v2 simulate endpoint with custom parameters
 */
export async function runFloodSimulation(
  request: SimulationRequest
): Promise<SimulationResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/flood/india/v2/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to run flood simulation:', error);
    throw error;
  }
}

/**
 * Check API health and component status
 * Returns connection status and model readiness
 */
export async function checkApiHealth(): Promise<HealthCheckResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
  
  try {
    const startTime = performance.now();
    const response = await fetch(`${API_BASE_URL}/api/flood/india/v2/health`, {
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      return {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        latency_ms: Math.round(performance.now() - startTime),
        components: {
          api: `error: HTTP ${response.status}`,
          random_forest: 'unknown',
          lstm: 'unknown',
          hydrological_model: 'unknown',
          rivers_database: 'unknown'
        },
        rivers_supported: [],
        version: 'unknown'
      };
    }
    
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error('API health check failed:', errorMessage);
    
    return {
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      latency_ms: -1,
      components: {
        api: `error: ${errorMessage}`,
        random_forest: 'unknown',
        lstm: 'unknown',
        hydrological_model: 'unknown',
        rivers_database: 'unknown'
      },
      rivers_supported: [],
      version: 'unknown'
    };
  }
}

/**
 * Get current model status and training info
 */
export async function getModelStatus(): Promise<ModelStatus> {
  const cacheKey = getCacheKey('model-status');
  const cached = getFromCache<ModelStatus>(cacheKey);
  if (cached) return cached;

  try {
    // Use v2 frontend-optimized API endpoint
    const response = await fetch(`${API_BASE_URL}/api/flood/india/v2/model-status`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    
    // v2 API already returns data in the correct format
    setCache(cacheKey, data);
    return data;
  } catch (error) {
    console.error('Failed to fetch model status:', error);
    throw error;
  }
}

/**
 * Trigger model retraining for a specific river
 */
export async function retrainModel(riverId: string): Promise<{
  success: boolean;
  message: string;
  metrics?: {
    accuracy: number;
    f1_score: number;
  };
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/flood/india/train/${riverId}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    // Clear cache after retraining
    clearIndiaFloodCache();
    
    return await response.json();
  } catch (error) {
    console.error(`Failed to retrain model for river ${riverId}:`, error);
    throw error;
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get risk level color for UI display
 */
export function getRiskColor(risk: string): string {
  const colors: Record<string, string> = {
    low: '#10b981',      // Green
    moderate: '#f59e0b', // Yellow
    high: '#ef4444',     // Red
    critical: '#7c3aed', // Purple
  };
  return colors[risk.toLowerCase()] || '#6b7280';
}

/**
 * Get trend icon direction
 */
export function getTrendDirection(trend: string): 'up' | 'down' | 'right' {
  switch (trend.toLowerCase()) {
    case 'rising':
      return 'up';
    case 'falling':
      return 'down';
    default:
      return 'right';
  }
}

/**
 * Format water level for display
 */
export function formatWaterLevel(level: number): string {
  return `${level.toFixed(2)} m`;
}

/**
 * Format probability as percentage
 */
export function formatProbability(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}

/**
 * Calculate time to danger level
 */
export function getTimeToAlert(
  currentLevel: number,
  predictions: WaterLevelPrediction[],
  dangerLevel: number
): string {
  const dangerPrediction = predictions.find(p => p.predicted_level >= dangerLevel);
  if (!dangerPrediction) {
    return 'No danger predicted in forecast period';
  }
  return `${dangerPrediction.hour} hours`;
}

const indiaFloodApi = {
  getIndiaRivers,
  getRiverPrediction,
  getCustomPrediction,
  runFloodSimulation,
  checkApiHealth,
  getModelStatus,
  retrainModel,
  clearIndiaFloodCache,
  getRiskColor,
  getTrendDirection,
  formatWaterLevel,
  formatProbability,
  getTimeToAlert,
};

export default indiaFloodApi;
