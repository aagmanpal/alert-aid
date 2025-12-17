/**
 * SHARED LOCATION DATA INTERFACES
 * Centralized location types for consistent usage across components
 */

export interface LocationData {
  latitude: number;
  longitude: number;
  city: string;
  state: string;
  country: string;
  timestamp: number;
  address?: string;
  accuracy?: number;
  isManual?: boolean;
}

export interface ReverseGeocodeResponse {
  city: string;
  state: string;
  country: string;
}

export interface LocationPermissionModalProps {
  isOpen: boolean;
  onLocationGranted: (location: LocationData) => void;
  onLocationDenied?: () => void;
  onManualEntry?: () => void;
  onClose?: () => void;
}

export interface GeolocationContextType {
  currentLocation: LocationData | null;
  isLoading: boolean;
  error: string | null;
  requestLocation: () => Promise<void>;
  clearError: () => void;
}