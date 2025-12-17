// Location Resolution Service
// Uses OpenWeatherMap Geocoding API for accurate location resolution

export interface LocationDetails {
  city: string;
  state: string;
  country: string;
  latitude: number;
  longitude: number;
  fullAddress: string;
  postalCode?: string;
  county?: string;
}

export interface OpenWeatherMapGeocodeResponse {
  name: string;
  local_names?: Record<string, string>;
  lat: number;
  lon: number;
  country: string;
  state?: string;
}

class LocationResolver {
  private static readonly OPENWEATHER_GEOCODE_URL = 'http://api.openweathermap.org/geo/1.0/reverse';
  private static readonly API_KEY = '1801423b3942e324ab80f5b47afe0859'; // Your API key
  private static readonly CACHE_DURATION = 1000 * 60 * 15; // 15 minutes (shorter cache)
  private static cache = new Map<string, { data: LocationDetails; timestamp: number }>();

  /**
   * Reverse geocode coordinates to get human-readable location details using OpenWeatherMap
   */
  static async getLocationDetails(lat: number, lon: number): Promise<LocationDetails> {
    const cacheKey = `${lat.toFixed(4)},${lon.toFixed(4)}`;
    
    // Check cache first
    const cached = this.cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < this.CACHE_DURATION) {
      console.log('🏷️ Using cached location data for', cacheKey);
      return cached.data;
    }

    try {
      console.log('🌍 Resolving location using OpenWeatherMap Geocoding API:', lat, lon);
      
      const response = await fetch(
        `${this.OPENWEATHER_GEOCODE_URL}?lat=${lat}&lon=${lon}&limit=1&appid=${this.API_KEY}`
      );

      if (!response.ok) {
        throw new Error(`OpenWeatherMap Geocoding API error: ${response.status}`);
      }

      const data: OpenWeatherMapGeocodeResponse[] = await response.json();
      
      if (!data || data.length === 0) {
        throw new Error('No location data found from OpenWeatherMap API');
      }

      const locationDetails = this.parseOpenWeatherMapResponse(data[0], lat, lon);
      
      // Cache the result
      this.cache.set(cacheKey, {
        data: locationDetails,
        timestamp: Date.now()
      });

      console.log('✅ Location resolved using OpenWeatherMap:', locationDetails);
      return locationDetails;

    } catch (error) {
      console.error('❌ Failed to resolve location with OpenWeatherMap:', error);
      
      // Return fallback with coordinates only
      return {
        city: 'Unknown City',
        state: 'Unknown State', 
        country: 'Unknown Country',
        latitude: lat,
        longitude: lon,
        fullAddress: `${lat.toFixed(4)}, ${lon.toFixed(4)}`,
      };
    }
  }

  /**
   * Parse OpenWeatherMap Geocoding API response to extract location components
   */
  private static parseOpenWeatherMapResponse(data: OpenWeatherMapGeocodeResponse, lat: number, lon: number): LocationDetails {
    // Extract location components from OpenWeatherMap response
    const city = data.name || 'Unknown City';
    const state = data.state || '';
    const country = data.country || 'Unknown Country';
    
    // Create formatted address
    let fullAddress: string;
    if (state) {
      fullAddress = `${city}, ${state}, ${country}`;
    } else {
      fullAddress = `${city}, ${country}`;
    }

    return {
      city,
      state,
      country,
      latitude: lat,
      longitude: lon,
      fullAddress
    };
  }

  /**
   * Get formatted location string for display
   */
  static formatLocationString(location: LocationDetails, format: 'short' | 'medium' | 'full' = 'medium'): string {
    switch (format) {
      case 'short':
        return `${location.city}, ${location.country}`;
      case 'full':
        return location.fullAddress;
      case 'medium':
      default:
        return location.state ? `${location.city}, ${location.state}, ${location.country}` : `${location.city}, ${location.country}`;
    }
  }

  /**
   * Validate coordinates are within valid ranges
   */
  static validateCoordinates(lat: number, lon: number): boolean {
    return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
  }

  /**
   * Clear location cache (useful for testing)
   */
  static clearCache(): void {
    this.cache.clear();
    console.log('🗑️ Location cache cleared');
  }
}

export default LocationResolver;