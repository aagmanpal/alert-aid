/**
 * TEMPERATURE CONVERSION UTILITIES
 * Provides standardized temperature handling and conversion functions
 * Centralizes temperature unit logic for consistent display
 */

export interface TemperatureData {
  value: number;
  unit: 'C' | 'F';
}

export class TemperatureConverter {
  /**
   * Convert Celsius to Fahrenheit
   */
  static celsiusToFahrenheit(celsius: number): number {
    return (celsius * 9/5) + 32;
  }

  /**
   * Convert Fahrenheit to Celsius
   */
  static fahrenheitToCelsius(fahrenheit: number): number {
    return (fahrenheit - 32) * 5/9;
  }

  /**
   * Format temperature with proper unit display
   */
  static formatTemperature(
    temp: number, 
    inputUnit: 'C' | 'F' = 'C', 
    outputUnit: 'C' | 'F' = 'C',
    precision: number = 1
  ): string {
    let convertedTemp = temp;

    // Convert if units differ
    if (inputUnit !== outputUnit) {
      if (inputUnit === 'C' && outputUnit === 'F') {
        convertedTemp = this.celsiusToFahrenheit(temp);
      } else if (inputUnit === 'F' && outputUnit === 'C') {
        convertedTemp = this.fahrenheitToCelsius(temp);
      }
    }

    return `${convertedTemp.toFixed(precision)}°${outputUnit}`;
  }

  /**
   * Standardize temperature to Celsius (our app standard)
   * Accepts temperature in any unit and returns Celsius
   */
  static standardizeToCelsius(temp: number, inputUnit: 'C' | 'F' = 'C'): number {
    if (inputUnit === 'F') {
      return this.fahrenheitToCelsius(temp);
    }
    return temp;
  }

  /**
   * Get user-preferred temperature format
   * In the future, this could read from user settings
   */
  static getPreferredUnit(): 'C' | 'F' {
    // For now, default to Celsius since that's what our APIs return
    // This can be made configurable in user settings later
    return 'C';
  }

  /**
   * Format temperature for display using user preference
   */
  static formatForDisplay(temp: number, inputUnit: 'C' | 'F' = 'C'): string {
    const preferredUnit = this.getPreferredUnit();
    return this.formatTemperature(temp, inputUnit, preferredUnit, 1);
  }

  /**
   * Validate temperature is within reasonable range
   */
  static isValidTemperature(temp: number, unit: 'C' | 'F' = 'C'): boolean {
    if (unit === 'C') {
      return temp >= -60 && temp <= 60; // Celsius range
    } else {
      return temp >= -76 && temp <= 140; // Fahrenheit range
    }
  }

  /**
   * Get temperature with range safety check
   */
  static safeTemperature(temp: number, unit: 'C' | 'F' = 'C', fallback: number = 20): number {
    if (this.isValidTemperature(temp, unit)) {
      return temp;
    }
    console.warn(`⚠️ Invalid temperature detected: ${temp}°${unit}, using fallback: ${fallback}°${unit}`);
    return fallback;
  }

  /**
   * Generate realistic temperature range based on location
   */
  static getLocationBasedTemperature(lat: number, season?: string): TemperatureData {
    // Base temperature calculation (same logic as backend)
    const baseTemp = 20 - Math.abs(lat) * 0.5;
    
    // Seasonal adjustments
    const seasonalOffset = {
      spring: 0,
      summer: 8,
      autumn: -2,
      winter: -10
    };

    const currentSeason = season || this.getCurrentSeason(lat > 0);
    const offset = seasonalOffset[currentSeason as keyof typeof seasonalOffset] || 0;
    
    const temperature = baseTemp + offset + (Math.random() - 0.5) * 10;
    
    return {
      value: Math.round(temperature * 10) / 10, // Round to 1 decimal
      unit: 'C'
    };
  }

  /**
   * Get current season based on hemisphere
   */
  private static getCurrentSeason(isNorthern: boolean): string {
    const month = new Date().getMonth() + 1; // 1-12
    
    if (isNorthern) {
      if (month >= 3 && month <= 5) return 'spring';
      if (month >= 6 && month <= 8) return 'summer';
      if (month >= 9 && month <= 11) return 'autumn';
      return 'winter';
    } else {
      // Southern hemisphere seasons are opposite
      if (month >= 3 && month <= 5) return 'autumn';
      if (month >= 6 && month <= 8) return 'winter';
      if (month >= 9 && month <= 11) return 'spring';
      return 'summer';
    }
  }

  /**
   * Batch convert array of temperatures
   */
  static convertTemperatureArray(
    temperatures: number[], 
    inputUnit: 'C' | 'F' = 'C', 
    outputUnit: 'C' | 'F' = 'C'
  ): number[] {
    return temperatures.map(temp => {
      if (inputUnit === outputUnit) return temp;
      return inputUnit === 'C' ? this.celsiusToFahrenheit(temp) : this.fahrenheitToCelsius(temp);
    });
  }
}

// Export commonly used functions for convenience
export const { 
  formatTemperature, 
  formatForDisplay, 
  celsiusToFahrenheit, 
  fahrenheitToCelsius, 
  standardizeToCelsius,
  safeTemperature
} = TemperatureConverter;

export default TemperatureConverter;