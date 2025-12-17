/**
 * PRODUCTION NAVIGATION BAR
 * 64px compact navigation with enhanced location service integration
 */

import React, { useState, useCallback, useEffect } from 'react';
import styled from 'styled-components';
import { 
  Home, 
  BarChart3, 
  AlertTriangle, 
  MapPin, 
  Shield,
  RefreshCw,
  Menu,
  X,
  Search
} from 'lucide-react';
import ManualLocationSearch from '../Location/ManualLocationSearch';
import { 
  productionColors, 
  productionAnimations
} from '../../styles/production-ui-system';
import { enhancedLocationService } from '../../services/enhancedLocationService';
import { LiveStatusBadge, GPSStatusBadge, LastUpdatedBadge } from '../common/StatusBadges';

const NavigationContainer = styled.nav`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  height: 64px; /* Updated to 64px for better visibility */
  background: linear-gradient(135deg, ${productionColors.background.secondary} 0%, ${productionColors.background.tertiary} 100%);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid ${productionColors.border.primary};
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  
  /* Enhanced shadow system */
  box-shadow: 
    0 2px 8px rgba(0, 0, 0, 0.15),
    0 1px 3px rgba(0, 0, 0, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.03);
`;

const Brand = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  color: ${productionColors.text.primary};
  font-weight: 700;
  font-size: 0.95rem;
  flex-shrink: 0;
  letter-spacing: 0.5px;
  
  .brand-logo {
    width: 36px;
    height: 36px;
    object-fit: contain;
    flex-shrink: 0;
    filter: drop-shadow(0 2px 8px rgba(239, 68, 68, 0.3));
  }
  
  span {
    background: linear-gradient(135deg, ${productionColors.text.primary} 0%, ${productionColors.brand.primary} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
`;

const NavigationItems = styled.div<{ isOpen: boolean }>`
  display: flex;
  align-items: center;
  gap: 8px;
  
  @media (max-width: 768px) {
    position: absolute;
    top: 64px;
    left: 0;
    right: 0;
    background: ${productionColors.background.secondary};
    border-bottom: 1px solid ${productionColors.border.primary};
    flex-direction: column;
    padding: 16px 24px;
    gap: 12px;
    transform: translateY(${({ isOpen }) => isOpen ? '0' : '-100%'});
    opacity: ${({ isOpen }) => isOpen ? '1' : '0'};
    visibility: ${({ isOpen }) => isOpen ? 'visible' : 'hidden'};
    transition: all ${productionAnimations.duration.normal} ${productionAnimations.easing.smooth};
  }
`;

const NavItem = styled.button<{ isActive?: boolean }>`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 0.8125rem;
  font-weight: 500;
  text-decoration: none;
  border: 1px solid transparent;
  cursor: pointer;
  white-space: nowrap;
  transition: all ${productionAnimations.duration.normal} ${productionAnimations.easing.smooth};
  position: relative;
  overflow: hidden;
  
  /* Aesthetic gradient background for active state */
  background: ${({ isActive }) => 
    isActive 
      ? `linear-gradient(135deg, ${productionColors.brand.primary} 0%, ${productionColors.brand.secondary} 100%)`
      : 'transparent'
  };
  color: ${({ isActive }) => 
    isActive 
      ? 'white' 
      : productionColors.text.secondary
  };
  
  /* Subtle glow effect for active button */
  box-shadow: ${({ isActive }) => 
    isActive 
      ? '0 4px 12px rgba(239, 68, 68, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
      : 'none'
  };
  
  /* Hover effect with smooth transform */
  &:hover {
    background: ${({ isActive }) => 
      isActive 
        ? `linear-gradient(135deg, ${productionColors.brand.secondary} 0%, ${productionColors.brand.dark} 100%)`
        : `rgba(239, 68, 68, 0.08)`
    };
    color: ${({ isActive }) => 
      isActive 
        ? 'white' 
        : productionColors.text.primary
    };
    transform: translateY(-2px);
    box-shadow: ${({ isActive }) => 
      isActive 
        ? '0 6px 16px rgba(239, 68, 68, 0.4)'
        : '0 2px 8px rgba(0, 0, 0, 0.1)'
    };
  }
  
  &:active {
    transform: translateY(0);
  }
  
  svg {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
  }
  
  @media (max-width: 768px) {
    width: 100%;
    justify-content: flex-start;
    padding: 10px 14px;
    font-size: 0.875rem;
  }
`;

const LocationSection = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: rgba(239, 68, 68, 0.05);
  border: 1px solid rgba(239, 68, 68, 0.1);
  border-radius: 8px;
  color: ${productionColors.text.secondary};
  font-size: 0.75rem;
  flex-shrink: 0;
  transition: all ${productionAnimations.duration.normal} ease;
  
  &:hover {
    background: rgba(239, 68, 68, 0.08);
    border-color: rgba(239, 68, 68, 0.2);
  }
  
  svg {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
    color: ${productionColors.brand.primary};
  }
  
  @media (max-width: 768px) {
    order: -1;
    width: 100%;
    padding: 10px 12px;
    margin-bottom: 12px;
    border-bottom: 1px solid ${productionColors.border.secondary};
    font-size: 0.8125rem;
  }
`;

const StatusBadgeContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  
  @media (max-width: 768px) {
    order: -1;
    width: 100%;
    justify-content: flex-start;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }
`;

const LocationText = styled.span`
  max-width: 150px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  
  @media (max-width: 768px) {
    max-width: none;
  }
`;

const RefreshButton = styled.button<{ isLoading?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  border: none;
  background: ${productionColors.background.tertiary};
  color: ${productionColors.brand.primary};
  cursor: pointer;
  transition: all ${productionAnimations.duration.normal} ${productionAnimations.easing.smooth};
  flex-shrink: 0;
  
  &:hover {
    background: ${productionColors.brand.primary};
    color: white;
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
  }
  
  &:active {
    transform: scale(0.95);
  }
  
  svg {
    width: 14px;
    height: 14px;
    animation: ${({ isLoading }) => 
      isLoading ? 'spin 1s linear infinite' : 'none'
    };
  }
`;

const MobileMenuButton = styled.button`
  display: none;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 4px;
  border: none;
  background: transparent;
  color: ${productionColors.text.primary};
  cursor: pointer;
  transition: all ${productionAnimations.duration.fast} ${productionAnimations.easing.smooth};
  
  &:hover {
    background: ${productionColors.interactive.hover};
  }
  
  svg {
    width: 18px;
    height: 18px;
  }
  
  @media (max-width: 768px) {
    display: flex;
  }
`;

const StarToggleButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: ${productionColors.text.primary};
  cursor: pointer;
  transition: all ${productionAnimations.duration.fast} ${productionAnimations.easing.smooth};
  flex-shrink: 0;

  &:hover {
    background: ${productionColors.interactive.hover};
  }

  &:focus {
    outline: 2px solid ${productionColors.border.accent};
    outline-offset: 2px;
  }

  &:focus:not(:focus-visible) {
    outline: none;
  }
`;

interface NavigationBarProps {
  currentPage?: string;
  onNavigate?: (page: string) => void;
}

export const NavigationBar: React.FC<NavigationBarProps> = ({ 
  currentPage = 'dashboard',
  onNavigate 
}) => {
  const [locationString, setLocationString] = useState('Detecting location...');
  const [isLocationLoading, setIsLocationLoading] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [isGPSEnabled, setIsGPSEnabled] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [showManualLocationModal, setShowManualLocationModal] = useState(false);

  // Navigation items
  const navigationItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'predictions', label: 'Predictions', icon: AlertTriangle },
    { id: 'alerts', label: 'Alerts', icon: AlertTriangle },
    { id: 'evacuation', label: 'Evacuation Routes', icon: Shield },
  ];

  const updateLocation = useCallback(async () => {
    setIsLocationLoading(true);
    setIsLive(false); // Set to offline during update
    try {
      // CLEAR ALL CACHES to prevent stale location data
      console.log('🗑️ Clearing ALL location caches...');
      localStorage.removeItem('enhanced-location-cache');
      localStorage.removeItem('alertaid-location');
      
      // Use browser geolocation API directly for most accurate results
      if ('geolocation' in navigator) {
        console.log('📍 Requesting fresh GPS location...');
        
        const position = await new Promise<GeolocationPosition>((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(resolve, reject, {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0 // Force fresh position
          });
        });
        
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        
        console.log(`✅ GPS coordinates obtained: ${lat}, ${lon}`);
        
        // Reverse geocode using Nominatim (most accurate)
        let locationData: any;
        try {
          const nominatimResponse = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&addressdetails=1`,
            {
              headers: { 'User-Agent': 'AlertAid-DisasterApp/1.0' }
            }
          );
          
          const nominatimData = await nominatimResponse.json();
          if (nominatimData && nominatimData.address) {
            const addr = nominatimData.address;
            const cityName = addr.city || addr.town || addr.village || addr.suburb || 
                            addr.municipality || addr.county || 'Unknown';
            locationData = {
              latitude: lat,
              longitude: lon,
              city: cityName,
              state: addr.state || addr.state_district || '',
              country: addr.country || 'Unknown',
              address: nominatimData.display_name,
              source: 'gps',
              timestamp: Date.now(),
              isManual: false
            };
            console.log('✅ Nominatim geocode:', cityName);
          } else {
            throw new Error('Nominatim failed');
          }
        } catch (e) {
          // Fallback to OpenWeatherMap
          console.warn('Nominatim failed, using OpenWeatherMap');
          const response = await fetch(
            `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=1801423b3942e324ab80f5b47afe0859`
          );
          const data = await response.json();
          locationData = data && data.length > 0 ? {
            latitude: lat,
            longitude: lon,
            city: data[0].name || 'Unknown',
            state: data[0].state || '',
            country: data[0].country || 'Unknown',
            source: 'gps',
            timestamp: Date.now(),
            isManual: false
          } : {
            latitude: lat,
            longitude: lon,
            city: `${lat.toFixed(2)}°, ${lon.toFixed(2)}°`,
            state: '',
            country: 'GPS Location',
            source: 'gps',
            timestamp: Date.now(),
            isManual: false
          };
        }
        
        const displayString = `${locationData.city}, ${locationData.state || locationData.country}`;
        setLocationString(displayString);
        setIsGPSEnabled(true);
        setIsLive(true);
        setLastUpdated(new Date());
        
        // Save to localStorage
        localStorage.setItem('enhanced-location-cache', JSON.stringify(locationData));
        localStorage.setItem('alertaid-location', JSON.stringify(locationData));
        
        console.log('📍 Location updated to:', displayString, locationData);
        
        // Broadcast location change event
        console.log('📡 Broadcasting location-changed event...');
        window.dispatchEvent(new CustomEvent('location-changed', { detail: locationData }));
        
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('location-changed', { detail: locationData }));
        }, 150);
      } else {
        throw new Error('Geolocation not supported');
      }
    } catch (error) {
      console.error('Location update failed:', error);
      setLocationString('Location unavailable');
      setIsLive(false);
    } finally {
      setIsLocationLoading(false);
    }
  }, []);

  // Clear all location caches and force fresh GPS location
  const clearLocationCache = useCallback(() => {
    console.log('🧹 Clearing all location caches...');
    localStorage.removeItem('alertaid-location');
    localStorage.removeItem('enhanced-location-cache');
    localStorage.removeItem('alertaid-location-prompted');
    setLocationString('Refreshing...');
    setIsGPSEnabled(false);
    setIsLive(false);
    
    // Trigger fresh GPS location after clearing
    setTimeout(() => {
      updateLocation();
    }, 100);
  }, [updateLocation]);

  // Initialize location on mount (but don't auto-refresh - let LocationContext handle it)
  useEffect(() => {
    // Only update if we don't have a valid location in localStorage
    const savedLocation = localStorage.getItem('alertaid-location');
    if (!savedLocation) {
      console.log('📍 NavigationBar: No saved location, triggering update');
      updateLocation();
    } else {
      try {
        const locationData = JSON.parse(savedLocation);
        const displayString = `${locationData.city}, ${locationData.state || locationData.country}`;
        setLocationString(displayString);
        setIsGPSEnabled(locationData.source === 'gps');
        setIsLive(true);
        setLastUpdated(new Date(locationData.timestamp || Date.now()));
        console.log('📍 NavigationBar: Using saved location:', displayString);
      } catch (e) {
        console.warn('Failed to parse saved location, fetching fresh');
        updateLocation();
      }
    }
    
    // Listen for location changes from other components
    const handleLocationUpdate = (event: any) => {
      if (event.detail) {
        const locationData = event.detail;
        const displayString = `${locationData.city}, ${locationData.state || locationData.country}`;
        setLocationString(displayString);
        setIsGPSEnabled(locationData.source === 'gps');
        setIsLive(true);
        setLastUpdated(new Date(locationData.timestamp || Date.now()));
        console.log('📍 NavigationBar: Received location update:', displayString);
      }
    };
    
    window.addEventListener('location-changed', handleLocationUpdate);
    
    return () => {
      window.removeEventListener('location-changed', handleLocationUpdate);
    };
  }, [updateLocation]);

  const handleNavigation = useCallback((pageId: string) => {
    setIsMobileMenuOpen(false);
    onNavigate?.(pageId);
  }, [onNavigate]);

  const toggleMobileMenu = useCallback(() => {
    setIsMobileMenuOpen(prev => !prev);
  }, []);

  const handleManualLocationSelect = useCallback((location: { latitude: number; longitude: number; city?: string; country?: string }) => {
    // Update location display
    const displayString = location.city 
      ? `${location.city}${location.country ? `, ${location.country}` : ''}`
      : `${location.latitude.toFixed(4)}°, ${location.longitude.toFixed(4)}°`;
    
    setLocationString(displayString);
    setIsGPSEnabled(false); // Manual location, not GPS
    setIsLive(true);
    setLastUpdated(new Date());
    setShowManualLocationModal(false);
    
    // Save to localStorage for persistence in BOTH locations
    const locationData = {
      latitude: location.latitude,
      longitude: location.longitude,
      city: location.city || 'Custom Location',
      state: '',
      country: location.country || '',
      source: 'manual',
      timestamp: Date.now(),
      isManual: true
    };
    
    // Save to both cache locations
    localStorage.setItem('enhanced-location-cache', JSON.stringify(locationData));
    localStorage.setItem('alertaid-location', JSON.stringify(locationData));
    localStorage.setItem('alertaid-location-prompted', 'granted');
    
    console.log('📍 Manual location set and saved to localStorage:', locationData);
    
    // Dispatch event for ALL components to refresh their data
    console.log('📡 Broadcasting location-changed event...');
    window.dispatchEvent(new CustomEvent('location-changed', { detail: locationData }));
    
    // Small delay then dispatch again to ensure all listeners catch it
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('location-changed', { detail: locationData }));
    }, 100);
  }, []);

  return (
    <>
      {showManualLocationModal && (
        <ManualLocationSearch
          onLocationSelect={handleManualLocationSelect}
          onClose={() => setShowManualLocationModal(false)}
        />
      )}
    <NavigationContainer>
      <Brand>
        <img 
          src="/Gemini_Generated_Image_7c3uv87c3uv87c3u.svg" 
          alt="Alert Aid Logo" 
          className="brand-logo"
        />
        <span>Alert Aid</span>
      </Brand>

      <NavigationItems isOpen={isMobileMenuOpen}>
        <StatusBadgeContainer>
          <LiveStatusBadge isLive={isLive} />
          <GPSStatusBadge enabled={isGPSEnabled} />
          <LastUpdatedBadge timestamp={lastUpdated} />
        </StatusBadgeContainer>
        
        <LocationSection>
          <MapPin size={12} />
          <LocationText>{locationString}</LocationText>
          <RefreshButton 
            onClick={() => setShowManualLocationModal(true)}
            title="Search location manually"
          >
            <Search />
          </RefreshButton>
          <RefreshButton 
            onClick={updateLocation}
            isLoading={isLocationLoading}
            title="Refresh location"
          >
            <RefreshCw />
          </RefreshButton>
        </LocationSection>

        {navigationItems.map(({ id, label, icon: Icon }) => (
          <NavItem
            key={id}
            isActive={currentPage === id}
            onClick={() => handleNavigation(id)}
          >
            <Icon />
            <span>{label}</span>
          </NavItem>
        ))}
      </NavigationItems>

      <MobileMenuButton 
        onClick={toggleMobileMenu}
        aria-label="Toggle navigation menu"
      >
        {isMobileMenuOpen ? <X /> : <Menu />}
      </MobileMenuButton>
      <StarToggleButton
        title="Toggle starfield background"
        aria-label="Toggle starfield background"
        onClick={() => {
          try {
            const cur = localStorage.getItem('starfieldEnabled');
            const next = cur === null ? 'false' : (cur === 'true' ? 'false' : 'true');
            localStorage.setItem('starfieldEnabled', next);
            // emit global event
            window.dispatchEvent(new CustomEvent('starfield:toggle', { detail: next === 'true' }));
          } catch (err) {
            console.warn('Failed toggling starfield', err);
          }
        }}
      >
        {/* simple star icon using unicode to avoid adding icon deps */}
        <span style={{fontSize: 14}}>✦</span>
      </StarToggleButton>
    </NavigationContainer>
    </>
  );
};

export default NavigationBar;