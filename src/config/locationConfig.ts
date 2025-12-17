/**
 * LOCATION CONFIGURATION
 * Set default location for the application
 * 
 * PRODUCTION NOTE: GPS detection is ENABLED by default
 * Application will request user's real-time location
 * Fallback location is only used if GPS is denied/unavailable
 */

export const DEFAULT_LOCATION = {
  latitude: 26.9124,
  longitude: 75.7873,
  city: 'Jaipur',
  state: 'Rajasthan',
  country: 'India'
};

// Set to true to always use default location (ignore GPS)
// PRODUCTION: Set to FALSE to use real GPS location
export const FORCE_DEFAULT_LOCATION = false;

// Set to false to disable GPS detection entirely
// PRODUCTION: Set to TRUE to enable real GPS detection
export const ENABLE_GPS_DETECTION = true;

const locationConfig = {
  DEFAULT_LOCATION,
  FORCE_DEFAULT_LOCATION,
  ENABLE_GPS_DETECTION
};

export default locationConfig;
