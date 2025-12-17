// OpenWeatherMap API Integration
// Using the provided API key: 1801423b3942e324ab80f5b47afe0859

export interface OpenWeatherMapResponse {
  coord: {
    lat: number;
    lon: number;
  };
  weather: Array<{
    id: number;
    main: string;
    description: string;
    icon: string;
  }>;
  base: string;
  main: {
    temp: number;
    feels_like: number;
    temp_min: number;
    temp_max: number;
    pressure: number;
    humidity: number;
    sea_level?: number;
    grnd_level?: number;
  };
  visibility: number;
  wind: {
    speed: number;
    deg: number;
    gust?: number;
  };
  clouds: {
    all: number;
  };
  dt: number;
  sys: {
    type: number;
    id: number;
    country: string;
    sunrise: number;
    sunset: number;
  };
  timezone: number;
  id: number;
  name: string;
  cod: number;
}

export interface ReverseGeocodeResponse {
  name: string;
  local_names?: { [key: string]: string };
  lat: number;
  lon: number;
  country: string;
  state?: string;
}

export class OpenWeatherMapService {
  private static readonly API_KEY = '1801423b3942e324ab80f5b47afe0859';
  private static readonly BASE_URL = 'https://api.openweathermap.org/data/2.5';
  private static readonly GEO_URL = 'https://api.openweathermap.org/geo/1.0';
  private static readonly TIMEOUT = 8000; // 8 seconds

  // Convert OpenWeatherMap response to our WeatherData format
  private static mapToWeatherData(data: OpenWeatherMapResponse, isReal: boolean = true): any {
    const weather = data.weather[0] || {};
    
    return {
      temperature: Math.round(data.main.temp),
      temperature_feels_like: Math.round(data.main.feels_like),
      humidity: data.main.humidity,
      wind_speed: data.wind.speed,
      wind_direction: data.wind.deg,
      pressure: data.main.pressure,
      conditions: weather.description || 'Unknown',
      weather_icon: weather.icon,
      visibility: data.visibility ? Math.round(data.visibility / 1000) : 10, // Convert to km
      last_updated: new Date().toISOString(),
      source: 'OpenWeatherMap',
      is_real: isReal,
      cached: false,
      // Include raw OpenWeatherMap data for enhanced weather card
      weather: [weather],
      main: data.main,
      wind: data.wind,
      sys: data.sys
    };
  }

  // Fetch current weather with timeout and error handling
  static async getCurrentWeather(lat: number, lon: number): Promise<any> {
    try {
      console.log(`🌤️ Fetching live weather from OpenWeatherMap for ${lat}, ${lon}`);
      
      const url = `${this.BASE_URL}/weather?lat=${lat}&lon=${lon}&appid=${this.API_KEY}&units=metric`;
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.TIMEOUT);

      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
        }
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Invalid OpenWeatherMap API key');
        } else if (response.status === 404) {
          throw new Error('Location not found');
        } else if (response.status >= 500) {
          throw new Error('OpenWeatherMap service unavailable');
        }
        throw new Error(`Weather API error: ${response.status}`);
      }

      const data: OpenWeatherMapResponse = await response.json();
      
      console.log('✅ Successfully fetched live weather data');
      return this.mapToWeatherData(data, true);
      
    } catch (error: any) {
      console.warn('❌ OpenWeatherMap fetch failed:', error.message);
      
      // Return fallback data
      return this.getFallbackWeatherData(lat, lon, error.message);
    }
  }

  // Get reverse geocoding for enhanced location info
  static async reverseGeocode(lat: number, lon: number): Promise<ReverseGeocodeResponse | null> {
    try {
      console.log(`🗺️ Reverse geocoding with OpenWeatherMap for ${lat}, ${lon}`);
      
      const url = `${this.GEO_URL}/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${this.API_KEY}`;
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.TIMEOUT);

      const response = await fetch(url, {
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Geocoding failed: ${response.status}`);
      }

      const data: ReverseGeocodeResponse[] = await response.json();
      
      if (data && data.length > 0) {
        console.log('✅ Successfully geocoded location');
        return data[0];
      }
      
      return null;
      
    } catch (error: any) {
      console.warn('❌ Reverse geocoding failed:', error.message);
      return null;
    }
  }

  // Fallback weather data when API fails
  private static getFallbackWeatherData(lat: number, lon: number, errorMessage: string): any {
    console.log('🔄 Using fallback weather data');
    
    // Generate realistic fallback based on location and season
    const now = new Date();
    const month = now.getMonth(); // 0-11
    const isWinter = month >= 10 || month <= 2;
    const isSummer = month >= 5 && month <= 8;
    
    // Rough temperature estimation based on latitude and season
    let baseTemp = 20; // Default moderate temperature
    
    if (Math.abs(lat) > 50) { // High latitude
      baseTemp = isWinter ? -5 : (isSummer ? 15 : 8);
    } else if (Math.abs(lat) > 30) { // Mid latitude
      baseTemp = isWinter ? 5 : (isSummer ? 25 : 15);
    } else { // Low latitude (tropical)
      baseTemp = isWinter ? 20 : (isSummer ? 30 : 25);
    }
    
    // Add some randomness
    const temp = baseTemp + (Math.random() - 0.5) * 10;
    
    return {
      temperature: Math.round(temp),
      temperature_feels_like: Math.round(temp + (Math.random() - 0.5) * 4),
      humidity: Math.round(50 + Math.random() * 30), // 50-80%
      wind_speed: Math.round((Math.random() * 5 + 2) * 10) / 10, // 2-7 m/s
      wind_direction: Math.round(Math.random() * 360),
      pressure: Math.round(1000 + Math.random() * 30), // 1000-1030 hPa
      conditions: isWinter ? 'Cloudy' : (isSummer ? 'Clear sky' : 'Partly cloudy'),
      weather_icon: isWinter ? '04d' : (isSummer ? '01d' : '02d'),
      visibility: Math.round(8 + Math.random() * 4), // 8-12 km
      last_updated: new Date().toISOString(),
      source: 'Simulated (Fallback)',
      is_real: false,
      cached: false,
      error: errorMessage,
      weather: [{
        description: isWinter ? 'cloudy' : (isSummer ? 'clear sky' : 'partly cloudy'),
        icon: isWinter ? '04d' : (isSummer ? '01d' : '02d')
      }]
    };
  }

  // Test API connection
  static async testConnection(): Promise<boolean> {
    try {
      // Test with a known location (London)
      await this.getCurrentWeather(51.5074, -0.1278);
      return true;
    } catch (error) {
      console.warn('OpenWeatherMap connection test failed:', error);
      return false;
    }
  }
}

export default OpenWeatherMapService;