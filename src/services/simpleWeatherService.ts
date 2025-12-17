/**
 * SIMPLE WEATHER SERVICE
 * Direct OpenWeatherMap API integration for reliable weather data
 */

const OPENWEATHER_API_KEY = '1801423b3942e324ab80f5b47afe0859';
const ONE_CALL_API_URL = 'https://api.openweathermap.org/data/3.0/onecall';
const WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5/weather';

export interface SimpleWeatherData {
  current: {
    temp: number;
    feels_like: number;
    humidity: number;
    pressure: number;
    wind_speed: number;
    visibility: number;
    uvi: number;
    weather: Array<{
      main: string;
      description: string;
      icon: string;
    }>;
  };
  last_updated?: string;
  is_real: boolean;
  source: string;
}

export class SimpleWeatherService {
  /**
   * Fetch weather data using free 2.5 API (3.0 requires paid subscription)
   */
  static async getWeather(lat: number, lon: number): Promise<SimpleWeatherData> {
    console.log(`🌤️ [SimpleWeatherService] Fetching weather for ${lat}, ${lon}`);
    
    try {
      // Use Current Weather API 2.5 (free tier)
      return await this.getCurrentWeatherFallback(lat, lon);
    } catch (error) {
      console.error('❌ [SimpleWeatherService] Error:', error);
      throw error;
    }
  }

  /**
   * Fallback using current weather API 2.5
   */
  private static async getCurrentWeatherFallback(lat: number, lon: number): Promise<SimpleWeatherData> {
    console.log('🔄 [SimpleWeatherService] Using Current Weather API 2.5 fallback');
    
    try {
      const weatherUrl = `${WEATHER_API_URL}?lat=${lat}&lon=${lon}&appid=${OPENWEATHER_API_KEY}&units=metric`;
      console.log('🌐 [SimpleWeatherService] Fallback URL:', weatherUrl);
      
      const response = await fetch(weatherUrl);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('✅ [SimpleWeatherService] Fallback API success:', data);
      
      // Convert to our format
      const convertedData: SimpleWeatherData = {
        current: {
          temp: data.main.temp,
          feels_like: data.main.feels_like,
          humidity: data.main.humidity,
          pressure: data.main.pressure,
          wind_speed: data.wind.speed,
          visibility: data.visibility,
          uvi: 0, // Not available in current weather API
          weather: data.weather
        },
        last_updated: new Date().toISOString(),
        is_real: true,
        source: 'OpenWeatherMap Current Weather API 2.5'
      };
      
      console.log('✅ [SimpleWeatherService] Converted data:', convertedData);
      return convertedData;
      
    } catch (error) {
      console.error('❌ [SimpleWeatherService] Fallback failed:', error);
      
      // Ultimate fallback: reasonable default data for Jaipur
      return this.getDefaultWeatherData();
    }
  }

  /**
   * REMOVED: Default fallback data
   * Production system MUST use real API data only
   * If all APIs fail, component should show error state with retry button
   */
  private static getDefaultWeatherData(): SimpleWeatherData {
    console.error('❌ [SimpleWeatherService] ALL WEATHER APIs FAILED - NO FALLBACK DATA');
    
    throw new Error('Weather service unavailable. All API endpoints failed. Please check your internet connection and try again.');
  }
}

export default SimpleWeatherService;
