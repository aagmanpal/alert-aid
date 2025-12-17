import { useState, useEffect, useCallback } from 'react';

// Types for real-time disaster data
export interface WeatherData {
  temperature: number;
  humidity: number;
  pressure: number;
  windSpeed: number;
  windDirection: number;
  precipitation: number;
  visibility: number;
  uvIndex: number;
  condition: string;
  timestamp: number;
}

export interface EarthquakeData {
  magnitude: number;
  depth: number;
  latitude: number;
  longitude: number;
  location: string;
  timestamp: number;
  distance: number; // km from user location
  intensity: string;
}

export interface WildfireData {
  latitude: number;
  longitude: number;
  brightness: number;
  confidence: number;
  distance: number; // km from user location
  timestamp: number;
  severity: 'low' | 'medium' | 'high' | 'extreme';
}

export interface FloodWarning {
  severity: 'minor' | 'moderate' | 'major' | 'extreme';
  location: string;
  description: string;
  validUntil: number;
  affectedAreas: string[];
  evacuationRecommended: boolean;
}

export interface RiskAssessment {
  earthquake: number; // 0-10 scale
  flood: number;
  wildfire: number;
  storm: number;
  overall: number;
  factors: string[];
  lastUpdated: number;
}

// Custom hook for real-time disaster data
export const useRealTimeDisasterData = (latitude?: number, longitude?: number) => {
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [earthquakes, setEarthquakes] = useState<EarthquakeData[]>([]);
  const [wildfires, setWildfires] = useState<WildfireData[]>([]);
  const [floodWarnings, setFloodWarnings] = useState<FloodWarning[]>([]);
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // API Keys (in production, these should be environment variables)
  const OPENWEATHER_API_KEY = process.env.REACT_APP_OPENWEATHER_API_KEY || 'demo_key';
  
  // Calculate distance between two coordinates (Haversine formula)
  const calculateDistance = useCallback((lat1: number, lon1: number, lat2: number, lon2: number): number => {
    const R = 6371; // Earth's radius in kilometers
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  }, []);

  // Fetch current weather data from OpenWeatherMap
  const fetchWeatherData = useCallback(async (lat: number, lon: number) => {
    try {
      const response = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${OPENWEATHER_API_KEY}&units=metric`
      );
      const data = await response.json();
      
      if (response.ok) {
        const weather: WeatherData = {
          temperature: data.main.temp,
          humidity: data.main.humidity,
          pressure: data.main.pressure,
          windSpeed: data.wind?.speed || 0,
          windDirection: data.wind?.deg || 0,
          precipitation: data.rain?.['1h'] || data.snow?.['1h'] || 0,
          visibility: data.visibility || 10000,
          uvIndex: 0, // Requires separate API call
          condition: data.weather[0].main,
          timestamp: Date.now()
        };
        setWeatherData(weather);
      } else {
        throw new Error(data.message || 'Weather data fetch failed');
      }
    } catch (error) {
      console.error('Weather fetch error:', error);
      setError('Failed to fetch weather data');
    }
  }, [OPENWEATHER_API_KEY]);

  // Fetch earthquake data from USGS
  const fetchEarthquakeData = useCallback(async (lat: number, lon: number) => {
    try {
      // Get earthquakes from the past 24 hours within 500km
      const response = await fetch(
        'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson'
      );
      const data = await response.json();
      
      if (response.ok && data.features) {
        const nearbyEarthquakes: EarthquakeData[] = data.features
          .map((feature: any) => {
            const [lon, lat, depth] = feature.geometry.coordinates;
            const distance = calculateDistance(lat, lon, lat, lon);
            
            return {
              magnitude: feature.properties.mag,
              depth: depth || 0,
              latitude: lat,
              longitude: lon,
              location: feature.properties.place,
              timestamp: feature.properties.time,
              distance,
              intensity: feature.properties.mag > 7 ? 'severe' : 
                       feature.properties.mag > 5 ? 'moderate' : 'minor'
            };
          })
          .filter((eq: EarthquakeData) => eq.distance <= 500) // Within 500km
          .sort((a: EarthquakeData, b: EarthquakeData) => a.distance - b.distance)
          .slice(0, 10); // Top 10 closest
        
        setEarthquakes(nearbyEarthquakes);
      }
    } catch (error) {
      console.error('Earthquake fetch error:', error);
      setError('Failed to fetch earthquake data');
    }
  }, [calculateDistance]);

  // Fetch wildfire data from NASA FIRMS
  const fetchWildfireData = useCallback(async (lat: number, lon: number) => {
    try {
      // NASA FIRMS API for active fires (last 24 hours)
      const response = await fetch(
        `https://firms.modaps.eosdis.nasa.gov/api/country/csv/7d559460871af0e2f75fe3a9a044a5b3/VIIRS_SNPP_NRT/USA/1`
      );
      const csvText = await response.text();
      
      if (response.ok) {
        const lines = csvText.split('\n');
        const headers = lines[0].split(',');
        
        const fires: WildfireData[] = lines
          .slice(1)
          .filter(line => line.trim())
          .map(line => {
            const values = line.split(',');
            const fireLat = parseFloat(values[0]);
            const fireLon = parseFloat(values[1]);
            const brightness = parseFloat(values[2]);
            const confidence = parseFloat(values[9]);
            const distance = calculateDistance(lat, lon, fireLat, fireLon);
            
            return {
              latitude: fireLat,
              longitude: fireLon,
              brightness,
              confidence,
              distance,
              timestamp: Date.now(),
              severity: (brightness > 400 ? 'extreme' :
                       brightness > 350 ? 'high' :
                       brightness > 300 ? 'medium' : 'low') as 'low' | 'medium' | 'high' | 'extreme'
            };
          })
          .filter(fire => fire.distance <= 100) // Within 100km
          .sort((a, b) => a.distance - b.distance)
          .slice(0, 20);
        
        setWildfires(fires);
      }
    } catch (error) {
      console.error('Wildfire fetch error:', error);
      // Use mock data for development
      setWildfires([]);
    }
  }, [calculateDistance]);

  // Fetch flood warnings (mock implementation - replace with real API)
  const fetchFloodWarnings = useCallback(async (lat: number, lon: number) => {
    try {
      // In production, integrate with NOAA/NWS APIs
      // For now, using mock data based on weather conditions
      const mockWarnings: FloodWarning[] = [];
      
      if (weatherData && weatherData.precipitation > 10) {
        mockWarnings.push({
          severity: weatherData.precipitation > 25 ? 'major' : 'moderate',
          location: 'Current Area',
          description: `Heavy rainfall detected (${weatherData.precipitation}mm/h). Flash flooding possible.`,
          validUntil: Date.now() + (6 * 60 * 60 * 1000), // 6 hours
          affectedAreas: ['Urban areas', 'Low-lying regions'],
          evacuationRecommended: weatherData.precipitation > 25
        });
      }
      
      setFloodWarnings(mockWarnings);
    } catch (error) {
      console.error('Flood warning fetch error:', error);
    }
  }, [weatherData]);

  // Calculate risk assessment using ML-like algorithm
  const calculateRiskAssessment = useCallback(async () => {
    if (!weatherData) return;

    try {
      // Earthquake risk calculation
      const recentEarthquakes = earthquakes.filter(eq => 
        Date.now() - eq.timestamp < 7 * 24 * 60 * 60 * 1000 // Last 7 days
      );
      const earthquakeRisk = Math.min(10, recentEarthquakes.reduce((risk, eq) => {
        const distanceFactor = Math.max(0, 1 - (eq.distance / 500));
        const magnitudeFactor = Math.min(1, eq.magnitude / 10);
        return risk + (magnitudeFactor * distanceFactor * 3);
      }, 0));

      // Flood risk calculation
      const floodRisk = Math.min(10, 
        (weatherData.precipitation / 10) * 3 +
        (weatherData.humidity / 100) * 2 +
        (floodWarnings.length > 0 ? 5 : 0)
      );

      // Wildfire risk calculation
      const wildfireRisk = Math.min(10,
        (wildfires.length / 5) * 4 +
        (weatherData.temperature > 35 ? 3 : 0) +
        (weatherData.humidity < 30 ? 2 : 0) +
        (weatherData.windSpeed > 15 ? 2 : 0)
      );

      // Storm risk calculation
      const stormRisk = Math.min(10,
        (weatherData.windSpeed / 20) * 4 +
        (weatherData.pressure < 1000 ? 3 : 0) +
        (weatherData.precipitation > 5 ? 3 : 0)
      );

      // Overall risk (weighted average)
      const overallRisk = (earthquakeRisk * 0.3 + floodRisk * 0.25 + wildfireRisk * 0.25 + stormRisk * 0.2);

      const factors = [];
      if (earthquakeRisk > 5) factors.push('Recent seismic activity');
      if (floodRisk > 5) factors.push('Heavy rainfall conditions');
      if (wildfireRisk > 5) factors.push('High fire danger');
      if (stormRisk > 5) factors.push('Severe weather conditions');

      const assessment: RiskAssessment = {
        earthquake: Math.round(earthquakeRisk * 10) / 10,
        flood: Math.round(floodRisk * 10) / 10,
        wildfire: Math.round(wildfireRisk * 10) / 10,
        storm: Math.round(stormRisk * 10) / 10,
        overall: Math.round(overallRisk * 10) / 10,
        factors,
        lastUpdated: Date.now()
      };

      setRiskAssessment(assessment);
    } catch (error) {
      console.error('Risk assessment calculation error:', error);
    }
  }, [weatherData, earthquakes, wildfires, floodWarnings]);

  // Main data fetching function
  const fetchAllData = useCallback(async () => {
    if (!latitude || !longitude) return;

    setIsLoading(true);
    setError(null);

    try {
      // Fetch all data in parallel
      await Promise.all([
        fetchWeatherData(latitude, longitude),
        fetchEarthquakeData(latitude, longitude),
        fetchWildfireData(latitude, longitude)
      ]);
    } catch (error) {
      console.error('Data fetching error:', error);
      setError('Failed to fetch disaster data');
    } finally {
      setIsLoading(false);
    }
  }, [latitude, longitude, fetchWeatherData, fetchEarthquakeData, fetchWildfireData]);

  // Fetch flood warnings when weather data changes
  useEffect(() => {
    if (latitude && longitude) {
      fetchFloodWarnings(latitude, longitude);
    }
  }, [latitude, longitude, weatherData, fetchFloodWarnings]);

  // Calculate risk assessment when data changes
  useEffect(() => {
    calculateRiskAssessment();
  }, [calculateRiskAssessment]);

  // Initial data fetch
  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);

  // Set up periodic data refresh (every 5 minutes for weather, 15 minutes for others)
  useEffect(() => {
    if (!latitude || !longitude) return;

    const weatherInterval = setInterval(() => {
      fetchWeatherData(latitude, longitude);
    }, 5 * 60 * 1000); // 5 minutes

    const disasterInterval = setInterval(() => {
      fetchEarthquakeData(latitude, longitude);
      fetchWildfireData(latitude, longitude);
    }, 15 * 60 * 1000); // 15 minutes

    return () => {
      clearInterval(weatherInterval);
      clearInterval(disasterInterval);
    };
  }, [latitude, longitude, fetchWeatherData, fetchEarthquakeData, fetchWildfireData]);

  return {
    weatherData,
    earthquakes,
    wildfires,
    floodWarnings,
    riskAssessment,
    isLoading,
    error,
    refreshData: fetchAllData
  };
};