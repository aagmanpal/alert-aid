// API Service Layer for ML Model Integration
// This file provides structured endpoints for Python ML model communication

import React from 'react';

export interface MLPredictionRequest {
  sensorData: {
    seismic: number[];
    temperature: number[];
    humidity: number[];
    pressure: number[];
    windSpeed: number[];
    windDirection: number[];
  };
  location: {
    latitude: number;
    longitude: number;
    elevation: number;
  };
  timeWindow: {
    start: string; // ISO timestamp
    end: string;   // ISO timestamp
  };
}

export interface MLPredictionResponse {
  prediction: {
    disasterType: 'earthquake' | 'hurricane' | 'flood' | 'wildfire' | 'volcano' | 'tornado';
    probability: number; // 0-1
    confidence: number;  // 0-1
    riskScore: number;   // 1-10
    timeToEvent: number; // hours
  };
  modelInfo: {
    modelName: string;
    version: string;
    accuracy: number;
    lastTrained: string;
  };
  metadata: {
    requestId: string;
    timestamp: string;
    processingTime: number; // milliseconds
  };
}

export interface RealTimeAlert {
  id: string;
  type: 'seismic_anomaly' | 'weather_anomaly' | 'flood_risk' | 'wildfire_risk' | 'volcanic_activity';
  severity: 'low' | 'medium' | 'high' | 'critical';
  location: {
    name: string;
    coordinates: {
      lat: number;
      lng: number;
    };
  };
  metrics: {
    confidence: number;
    riskScore: number;
    anomalyScore: number;
  };
  description: string;
  timestamp: string;
  modelSource: string;
}

export interface ModelPerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  lastEvaluated: string;
  datasetSize: number;
  modelVersion: string;
}

// API Configuration
const API_CONFIG = {
  // Default to the local ML backend used by the project during development
  baseUrl: process.env.REACT_APP_ML_API_URL || 'http://127.0.0.1:8000',
  endpoints: {
    // Align endpoints to the FastAPI backend implemented in backend/enhanced_main.py
    predict: '/predict/disaster-risk',
    realTimeAlerts: '/alerts/stream',
    modelMetrics: '/model/performance',
    historicalData: '/data/historical',
    // Backend exposes weather and earthquake endpoints using lat/lon path params
    sensorData: '/weather',
    healthCheck: '/health'
  },
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${process.env.REACT_APP_ML_API_KEY || 'dev-key'}`
  }
};

// Mock data generators for development
export const mockPredictionResponse = (request: MLPredictionRequest): MLPredictionResponse => ({
  prediction: {
    disasterType: ['earthquake', 'hurricane', 'flood', 'wildfire', 'volcano'][Math.floor(Math.random() * 5)] as any,
    probability: Math.random() * 0.8 + 0.1, // 0.1-0.9
    confidence: Math.random() * 0.3 + 0.7,  // 0.7-1.0
    riskScore: Math.floor(Math.random() * 8) + 2, // 2-10
    timeToEvent: Math.floor(Math.random() * 168) + 1 // 1-168 hours
  },
  modelInfo: {
    modelName: 'DisasterNet-v3.1',
    version: '3.1.2',
    accuracy: 0.94 + Math.random() * 0.05, // 0.94-0.99
    lastTrained: '2024-01-15T10:30:00Z'
  },
  metadata: {
    requestId: `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date().toISOString(),
    processingTime: Math.floor(Math.random() * 500) + 100 // 100-600ms
  }
});

export const mockRealTimeAlert = (): RealTimeAlert => ({
  id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
  type: ['seismic_anomaly', 'weather_anomaly', 'flood_risk', 'wildfire_risk'][Math.floor(Math.random() * 4)] as any,
  severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any,
  location: {
    // Use a set of region names but avoid defaulting to any single city such as New York
    name: ['California', 'Texas', 'Florida', 'Pacific Northwest', 'Midwest'][Math.floor(Math.random() * 5)],
    coordinates: {
      lat: 25 + Math.random() * 25, // 25-50
      lng: -125 + Math.random() * 50 // -125 to -75
    }
  },
  metrics: {
    confidence: Math.random() * 0.3 + 0.7, // 0.7-1.0
    riskScore: Math.floor(Math.random() * 8) + 2, // 2-10
    anomalyScore: Math.random() * 0.5 + 0.5 // 0.5-1.0
  },
  description: 'Anomalous pattern detected by ML model requiring immediate attention',
  timestamp: new Date().toISOString(),
  modelSource: 'AnomalyDetector-v2.0'
});

// Core API service class
export class MLApiService {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(config = API_CONFIG) {
    this.baseUrl = config.baseUrl;
    this.headers = config.headers;
  }

  // Health check endpoint
  async healthCheck(): Promise<{ status: string; timestamp: string; models: string[] }> {
    if (process.env.NODE_ENV === 'development') {
      return {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        models: ['DisasterNet-v3.1', 'AnomalyDetector-v2.0', 'WeatherML-v3.0']
      };
    }

    const response = await fetch(`${this.baseUrl}${API_CONFIG.endpoints.healthCheck}`, {
      method: 'GET',
      headers: this.headers
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }

  // Main prediction endpoint
  async predictDisaster(request: MLPredictionRequest): Promise<MLPredictionResponse> {
    // Try the real backend first (even in development). If the backend is unreachable or returns an error,
    // gracefully fall back to the in-memory mock so the UI remains responsive during offline development.
    const url = `${this.baseUrl}${API_CONFIG.endpoints.predict}`;

    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(request),
      });

      if (!resp.ok) {
        // Non-2xx response — log and fall back to mock
        console.warn(`ML backend responded with ${resp.status} ${resp.statusText}, falling back to mock.`);
        // Short delay for UX parity
        await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 400));
        return mockPredictionResponse(request);
      }

      return await resp.json();
    } catch (err) {
      // Network error or fetch failed — fallback to mock response
      console.warn('ML backend unreachable, using mock prediction response:', err);
      await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 400));
      return mockPredictionResponse(request);
    }
  }

  // Real-time alerts stream (WebSocket in production, polling in dev)
  async *streamRealTimeAlerts(): AsyncGenerator<RealTimeAlert, void, unknown> {
    if (process.env.NODE_ENV === 'development') {
      // Mock streaming with intervals
      while (true) {
        await new Promise(resolve => setTimeout(resolve, 5000 + Math.random() * 10000)); // 5-15s
        if (Math.random() > 0.3) { // 70% chance of alert
          yield mockRealTimeAlert();
        }
      }
    } else {
      // WebSocket implementation for production
      const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}${API_CONFIG.endpoints.realTimeAlerts}`);
      
      while (true) {
        const message = await new Promise<string>((resolve, reject) => {
          ws.onmessage = (event) => resolve(event.data);
          ws.onerror = (error) => reject(error);
        });
        
        try {
          const alert: RealTimeAlert = JSON.parse(message);
          yield alert;
        } catch (error) {
          console.error('Failed to parse alert message:', error);
        }
      }
    }
  }

  // Get model performance metrics
  async getModelMetrics(): Promise<ModelPerformanceMetrics[]> {
    if (process.env.NODE_ENV === 'development') {
      return [
        {
          accuracy: 0.94 + Math.random() * 0.05,
          precision: 0.91 + Math.random() * 0.07,
          recall: 0.89 + Math.random() * 0.09,
          f1Score: 0.90 + Math.random() * 0.08,
          rocAuc: 0.92 + Math.random() * 0.06,
          lastEvaluated: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
          datasetSize: Math.floor(Math.random() * 50000) + 100000,
          modelVersion: 'DisasterNet-v3.1'
        },
        {
          accuracy: 0.88 + Math.random() * 0.08,
          precision: 0.85 + Math.random() * 0.1,
          recall: 0.87 + Math.random() * 0.09,
          f1Score: 0.86 + Math.random() * 0.1,
          rocAuc: 0.89 + Math.random() * 0.08,
          lastEvaluated: new Date(Date.now() - Math.random() * 5 * 24 * 60 * 60 * 1000).toISOString(),
          datasetSize: Math.floor(Math.random() * 30000) + 80000,
          modelVersion: 'AnomalyDetector-v2.0'
        }
      ];
    }

    const response = await fetch(`${this.baseUrl}${API_CONFIG.endpoints.modelMetrics}`, {
      method: 'GET',
      headers: this.headers
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch model metrics: ${response.statusText}`);
    }

    return response.json();
  }

  // Get historical incident data
  async getHistoricalData(params: {
    startDate: string;
    endDate: string;
    disasterTypes?: string[];
    location?: { lat: number; lng: number; radius: number };
  }): Promise<any[]> {
    if (process.env.NODE_ENV === 'development') {
      // Mock historical data
      const data = [];
      const start = new Date(params.startDate);
      const end = new Date(params.endDate);
      const days = Math.floor((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
      
      for (let i = 0; i < days; i++) {
        const date = new Date(start.getTime() + i * 24 * 60 * 60 * 1000);
        data.push({
          date: date.toISOString().split('T')[0],
          incidents: Math.floor(Math.random() * 20) + 5,
          predictions: Math.floor(Math.random() * 25) + 5,
          accuracy: 0.85 + Math.random() * 0.12
        });
      }
      
      return data;
    }

    const queryParams = new URLSearchParams({
      start_date: params.startDate,
      end_date: params.endDate,
      ...(params.disasterTypes && { disaster_types: params.disasterTypes.join(',') }),
      ...(params.location && { 
        lat: params.location.lat.toString(),
        lng: params.location.lng.toString(),
        radius: params.location.radius.toString()
      })
    });

    const response = await fetch(`${this.baseUrl}${API_CONFIG.endpoints.historicalData}?${queryParams}`, {
      method: 'GET',
      headers: this.headers
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch historical data: ${response.statusText}`);
    }

    return response.json();
  }

  // Get latest sensor data
  async getLatestSensorData(location: { lat: number; lng: number }): Promise<any> {
    if (process.env.NODE_ENV === 'development') {
      return {
        location,
        timestamp: new Date().toISOString(),
        sensors: {
          seismic: Array.from({ length: 24 }, () => Math.random() * 2),
          temperature: Array.from({ length: 24 }, () => 15 + Math.random() * 20),
          humidity: Array.from({ length: 24 }, () => 30 + Math.random() * 40),
          pressure: Array.from({ length: 24 }, () => 1000 + Math.random() * 50),
          windSpeed: Array.from({ length: 24 }, () => Math.random() * 25),
          windDirection: Array.from({ length: 24 }, () => Math.random() * 360)
        }
      };
    }

    const response = await fetch(`${this.baseUrl}${API_CONFIG.endpoints.sensorData}?lat=${location.lat}&lng=${location.lng}`, {
      method: 'GET',
      headers: this.headers
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch sensor data: ${response.statusText}`);
    }

    return response.json();
  }
}

// Singleton instance
export const mlApiService = new MLApiService();

// React hooks for ML API integration
export const useMLPrediction = () => {
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const predict = async (request: MLPredictionRequest): Promise<MLPredictionResponse | null> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await mlApiService.predictDisaster(request);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return { predict, isLoading, error };
};

export const useRealTimeAlerts = () => {
  const [alerts, setAlerts] = React.useState<RealTimeAlert[]>([]);
  const [isConnected, setIsConnected] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    
    const startStream = async () => {
      setIsConnected(true);
      
      try {
        for await (const alert of mlApiService.streamRealTimeAlerts()) {
          if (cancelled) break;
          
          setAlerts(prev => [alert, ...prev.slice(0, 49)]); // Keep last 50 alerts
        }
      } catch (error) {
        console.error('Alert stream error:', error);
        setIsConnected(false);
      }
    };

    startStream();

    return () => {
      cancelled = true;
      setIsConnected(false);
    };
  }, []);

  return { alerts, isConnected };
};

// React is imported at the top of this module for hooks