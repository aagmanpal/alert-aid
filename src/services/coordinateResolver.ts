/**
 * COORDINATE CONSISTENCY SERVICE
 * Ensures all API calls use the same, validated coordinates
 * Fixes inconsistency between different location sources
 */

import { LocationData } from '../components/Location/types';

// Standard fallback coordinates (Jaipur, India - safe test location)
export const FALLBACK_COORDINATES = {
  latitude: 26.9124,
  longitude: 75.7873,
  city: 'Jaipur',
  state: 'Rajasthan', 
  country: 'India',
  timestamp: Date.now()
} as const;

// Validated coordinate interface
export interface ResolvedCoordinates {
  latitude: number;
  longitude: number;
  source: 'geolocation' | 'manual' | 'fallback';
  locationData?: LocationData;
  isValidated: boolean;
}

/**
 * Coordinate Resolver Service
 * Provides consistent, validated coordinates for all API calls
 */
export class CoordinateResolver {
  
  /**
   * Resolve coordinates from multiple possible sources with validation
   */
  static resolveCoordinates(
    geolocationData?: LocationData | null,
    manualLocation?: LocationData | null,
    useFallback: boolean = true
  ): ResolvedCoordinates {
    
    // Priority 1: Valid geolocation data
    if (geolocationData && this.isValidCoordinate(geolocationData.latitude, geolocationData.longitude)) {
      console.log('📍 Using geolocation coordinates:', {
        lat: geolocationData.latitude,
        lon: geolocationData.longitude,
        city: geolocationData.city
      });
      
      return {
        latitude: geolocationData.latitude,
        longitude: geolocationData.longitude,
        source: 'geolocation',
        locationData: geolocationData,
        isValidated: true
      };
    }
    
    // Priority 2: Valid manual location data
    if (manualLocation && this.isValidCoordinate(manualLocation.latitude, manualLocation.longitude)) {
      console.log('📍 Using manual coordinates:', {
        lat: manualLocation.latitude,
        lon: manualLocation.longitude,
        city: manualLocation.city
      });
      
      return {
        latitude: manualLocation.latitude,
        longitude: manualLocation.longitude,
        source: 'manual',
        locationData: manualLocation,
        isValidated: true
      };
    }
    
    // Priority 3: Fallback coordinates (if allowed)
    if (useFallback) {
      console.log('📍 Using fallback coordinates (Jaipur):', {
        lat: FALLBACK_COORDINATES.latitude,
        lon: FALLBACK_COORDINATES.longitude
      });
      
      return {
        latitude: FALLBACK_COORDINATES.latitude,
        longitude: FALLBACK_COORDINATES.longitude,
        source: 'fallback',
        locationData: FALLBACK_COORDINATES as LocationData,
        isValidated: true
      };
    }
    
    // No valid coordinates available
    throw new Error('No valid coordinates available for API calls');
  }
  
  /**
   * Get coordinates for weather API calls
   */
  static getWeatherCoordinates(
    geolocationData?: LocationData | null,
    manualLocation?: LocationData | null
  ): { lat: number; lon: number } {
    const resolved = this.resolveCoordinates(geolocationData, manualLocation);
    return {
      lat: resolved.latitude,
      lon: resolved.longitude
    };
  }
  
  /**
   * Get coordinates for ML prediction calls
   */
  static getMLCoordinates(
    geolocationData?: LocationData | null,
    manualLocation?: LocationData | null
  ): LocationData {
    const resolved = this.resolveCoordinates(geolocationData, manualLocation);
    
    // Return full LocationData for ML API
    return resolved.locationData || {
      latitude: resolved.latitude,
      longitude: resolved.longitude,
      city: 'Unknown City',
      state: 'Unknown State',
      country: 'Unknown Country',
      timestamp: Date.now()
    };
  }
  
  /**
   * Get coordinates for alerts API calls
   */
  static getAlertsCoordinates(
    geolocationData?: LocationData | null,
    manualLocation?: LocationData | null
  ): { lat: number; lon: number } {
    const resolved = this.resolveCoordinates(geolocationData, manualLocation);
    return {
      lat: resolved.latitude,
      lon: resolved.longitude
    };
  }
  
  /**
   * Validate if coordinates are within valid ranges
   */
  static isValidCoordinate(latitude: number, longitude: number): boolean {
    return (
      typeof latitude === 'number' &&
      typeof longitude === 'number' &&
      !isNaN(latitude) &&
      !isNaN(longitude) &&
      latitude >= -90 &&
      latitude <= 90 &&
      longitude >= -180 &&
      longitude <= 180
    );
  }
  
  /**
   * Check if two coordinate sets are the same (within tolerance)
   */
  static areCoordinatesEqual(
    coord1: { latitude: number; longitude: number },
    coord2: { latitude: number; longitude: number },
    tolerance: number = 0.001 // ~100m tolerance
  ): boolean {
    const latDiff = Math.abs(coord1.latitude - coord2.latitude);
    const lonDiff = Math.abs(coord1.longitude - coord2.longitude);
    
    return latDiff <= tolerance && lonDiff <= tolerance;
  }
  
  /**
   * Get human-readable coordinate summary
   */
  static getCoordinateSummary(resolved: ResolvedCoordinates): string {
    const { latitude, longitude, source, locationData } = resolved;
    const location = locationData ? `${locationData.city}, ${locationData.country}` : 'Unknown Location';
    
    return `${location} (${latitude.toFixed(4)}, ${longitude.toFixed(4)}) [${source}]`;
  }
  
  /**
   * Log coordinate usage for debugging
   */
  static logCoordinateUsage(
    apiName: string,
    resolved: ResolvedCoordinates,
    additionalContext?: Record<string, any>
  ): void {
    console.log(`🎯 ${apiName} using coordinates:`, {
      coordinates: {
        lat: resolved.latitude,
        lon: resolved.longitude
      },
      source: resolved.source,
      location: resolved.locationData?.city || 'Unknown',
      summary: this.getCoordinateSummary(resolved),
      ...additionalContext
    });
  }
}

export default CoordinateResolver;