/**
 * Air Quality Service
 * Fetches and processes Air Quality Index (AQI) data from backend
 */

import { logger } from '../utils/logger';

export interface AQIData {
  aqi: number;
  level: string;
  color: string;
  description: string;
  components: {
    pm2_5: number;
    pm10: number;
    no2: number;
    o3: number;
    so2: number;
    co: number;
  };
  timestamp: string;
  location: {
    latitude: number;
    longitude: number;
  };
  is_real: boolean;
}

// Use environment variable or fallback to localhost
const rawApiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_BASE_URL = rawApiUrl.endsWith('/') ? rawApiUrl.slice(0, -1) : rawApiUrl;

class AirQualityService {
  private cache: Map<string, { data: AQIData; timestamp: number }> = new Map();
  private readonly CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

  /**
   * Fetch air quality data for given coordinates
   */
  async getAirQuality(latitude: number, longitude: number): Promise<AQIData> {
    const cacheKey = `${latitude.toFixed(4)},${longitude.toFixed(4)}`;
    
    // Check cache
    const cached = this.cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < this.CACHE_DURATION) {
      logger.info('🌫️ Using cached AQI data');
      return cached.data;
    }

    try {
      logger.info(`🌫️ Fetching AQI data for ${latitude}, ${longitude}`);
      
      const response = await fetch(
        `${API_BASE_URL}/api/weather/air-quality/${latitude}/${longitude}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`AQI API error: ${response.status}`);
      }

      const data: AQIData = await response.json();
      
      // Cache the result
      this.cache.set(cacheKey, {
        data,
        timestamp: Date.now(),
      });

      logger.info(`✅ AQI data received: ${data.level} (${data.aqi})`);
      return data;
    } catch (error) {
      logger.error('❌ Failed to fetch AQI data:', error);
      
      // Return fallback data
      return this.getFallbackAQI(latitude, longitude);
    }
  }

  /**
   * Get health recommendations based on AQI level
   */
  getHealthRecommendations(aqi: number): string[] {
    const recommendations: Record<number, string[]> = {
      1: [
        'Air quality is good - Enjoy outdoor activities',
        'No health precautions needed',
      ],
      2: [
        'Air quality is acceptable for most people',
        'Unusually sensitive individuals should consider limiting prolonged outdoor activities',
      ],
      3: [
        'Sensitive groups should reduce prolonged outdoor exertion',
        'General public should limit prolonged outdoor exertion',
        'Consider wearing a mask if sensitive',
      ],
      4: [
        'Everyone should avoid prolonged outdoor exertion',
        'Sensitive groups should avoid all outdoor activities',
        'Wear a mask when going outside',
        'Keep windows closed',
      ],
      5: [
        '⚠️ Health alert: Stay indoors',
        'Avoid all outdoor activities',
        'Use air purifiers indoors',
        'Wear N95 mask if you must go outside',
        'Seek medical attention if experiencing symptoms',
      ],
    };

    return recommendations[aqi] || recommendations[3];
  }

  /**
   * Check if AQI level warrants an alert
   */
  shouldAlert(aqi: number): boolean {
    return aqi >= 4; // Alert for Poor and Very Poor levels
  }

  /**
   * Get pollution severity for disaster risk calculation
   */
  getPollutionRiskFactor(aqi: number): number {
    // Returns a risk factor from 0 to 1
    const riskMap: Record<number, number> = {
      1: 0.0,
      2: 0.1,
      3: 0.3,
      4: 0.6,
      5: 1.0,
    };
    return riskMap[aqi] || 0.3;
  }

  /**
   * Generate fallback AQI data
   */
  private getFallbackAQI(latitude: number, longitude: number): AQIData {
    return {
      aqi: 2,
      level: 'Fair',
      color: 'yellow',
      description: 'Air quality is acceptable',
      components: {
        pm2_5: 25.5,
        pm10: 38.2,
        no2: 45.3,
        o3: 65.8,
        so2: 15.2,
        co: 450.0,
      },
      timestamp: new Date().toISOString(),
      location: { latitude, longitude },
      is_real: false,
    };
  }

  /**
   * Clear cache (useful for manual refresh)
   */
  clearCache(): void {
    this.cache.clear();
    logger.info('🗑️ AQI cache cleared');
  }
}

export const airQualityService = new AirQualityService();
export default airQualityService;
