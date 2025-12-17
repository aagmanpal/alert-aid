/**
 * LOCATION OVERRIDE UTILITY
 * Use this to manually set a location for testing or demonstration purposes
 */

import logger from './logger';

export interface ManualLocation {
  latitude: number;
  longitude: number;
  city: string;
  state: string;
  country: string;
}

// Predefined locations for quick testing
export const DEMO_LOCATIONS = {
  jaipur: {
    latitude: 26.9124,
    longitude: 75.7873,
    city: 'Jaipur',
    state: 'Rajasthan',
    country: 'India'
  },
  newYork: {
    latitude: 40.7128,
    longitude: -74.0060,
    city: 'New York',
    state: 'New York',
    country: 'United States'
  },
  tokyo: {
    latitude: 35.6762,
    longitude: 139.6503,
    city: 'Tokyo',
    state: 'Tokyo',
    country: 'Japan'
  },
  london: {
    latitude: 51.5074,
    longitude: -0.1278,
    city: 'London',
    state: 'England',
    country: 'United Kingdom'
  },
  colima: {
    latitude: 19.25971,
    longitude: -103.68941,
    city: 'Colima',
    state: 'Colima',
    country: 'Mexico'
  }
};

// Set manual location override
export const setManualLocation = (location: ManualLocation | keyof typeof DEMO_LOCATIONS) => {
  try {
    let locationData: ManualLocation;
    
    if (typeof location === 'string') {
      locationData = DEMO_LOCATIONS[location];
      if (!locationData) {
        logger.error(`❌ Unknown location: ${location}`);
        logger.log('Available locations:', Object.keys(DEMO_LOCATIONS));
        return false;
      }
    } else {
      locationData = location;
    }
    
    const overrideData = {
      ...locationData,
      timestamp: Date.now(),
      source: 'manual' as const,
      isManual: true
    };
    
    localStorage.setItem('location-override', JSON.stringify(overrideData));
    localStorage.setItem('enhanced-location-cache', JSON.stringify(overrideData));
    
    logger.log('✅ Manual location set:', locationData.city, locationData.state);
    logger.log('🔄 Reload the page to apply changes');
    
    return true;
  } catch (error) {
    logger.error('❌ Failed to set manual location:', error);
    return false;
  }
};

// Clear manual location override
export const clearManualLocation = () => {
  try {
    localStorage.removeItem('location-override');
    localStorage.removeItem('enhanced-location-cache');
    logger.log('✅ Manual location cleared');
    logger.log('🔄 Reload the page to use GPS location');
    return true;
  } catch (error) {
    logger.error('❌ Failed to clear manual location:', error);
    return false;
  }
};

// Get current location override
export const getManualLocation = (): ManualLocation | null => {
  try {
    const override = localStorage.getItem('location-override');
    if (override) {
      return JSON.parse(override);
    }
    return null;
  } catch (error) {
    logger.error('❌ Failed to get manual location:', error);
    return null;
  }
};

// Check if using manual location
export const isUsingManualLocation = (): boolean => {
  return localStorage.getItem('location-override') !== null;
};

// Make available globally
if (typeof window !== 'undefined') {
  (window as any).setLocation = (location: string | ManualLocation) => {
    const result = setManualLocation(location as any);
    if (result) {
      setTimeout(() => window.location.reload(), 1000);
    }
  };
  
  (window as any).clearLocation = () => {
    clearManualLocation();
    setTimeout(() => window.location.reload(), 1000);
  };
  
  (window as any).showLocations = () => {
    logger.log('📍 Available demo locations:');
    Object.entries(DEMO_LOCATIONS).forEach(([key, loc]) => {
      logger.log(`  - ${key}: ${loc.city}, ${loc.state}, ${loc.country}`);
    });
    logger.log('\n💡 Usage:');
    logger.log('  setLocation("jaipur")     - Set location to Jaipur');
    logger.log('  setLocation("tokyo")      - Set location to Tokyo');
    logger.log('  clearLocation()           - Use GPS location');
    logger.log('  showLocations()           - Show this help');
  };
  
  logger.log('💡 Location Override Tool loaded! Type showLocations() for help');
}

// Export location override utilities
const locationOverride = {
  setManualLocation,
  clearManualLocation,
  getManualLocation,
  isUsingManualLocation,
  DEMO_LOCATIONS
};

export default locationOverride;
