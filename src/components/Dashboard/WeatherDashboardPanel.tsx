import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { 
  Cloud, 
  Sun, 
  CloudRain, 
  CloudSnow, 
  Zap, 
  Wind,
  Thermometer,
  Droplets,
  Eye,
  Gauge,
  AlertTriangle,
  CheckCircle,
  Activity,
  MapPin,
  Clock
} from 'lucide-react';
import { enhancedSpacing } from '../../styles/enhanced-design-system';
import { 
  productionColors, 
  productionAnimations, 
  productionCard
} from '../../styles/production-ui-system';
import CachedAPIService from '../../services/apiService';
import { enhancedLocationService } from '../../services/enhancedLocationService';
import type { EnhancedWeatherData } from '../../types/weather';

// ENHANCED WEATHER LAYOUT - Optimized information density
const WeatherContainer = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr; /* Main weather gets more space */
  gap: ${enhancedSpacing[5]}; /* 20px for tighter layout */
  width: 100%;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr 1fr; /* Equal columns on tablet */
  }
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr; /* Single column on mobile */
    gap: ${enhancedSpacing[4]};
  }
`;

const CurrentWeatherCard = styled.div`
  ${productionCard}
  display: flex;
  flex-direction: column;
  gap: ${enhancedSpacing[3]}; /* Tighter spacing for density */
  min-height: 320px; /* Consistent height */
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: ${productionColors.gradients.brand};
    z-index: 1;
  }
`;

const WeatherHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${enhancedSpacing[4]};
`;

const WeatherTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: ${productionColors.text.primary};
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const LocationBadge = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.2);
  border-radius: 6px;
  font-size: 0.75rem;
  color: #EF4444;
`;

const CurrentConditions = styled.div`
  display: flex;
  align-items: center;
  gap: ${enhancedSpacing[4]};
  padding: ${enhancedSpacing[3]} 0; /* Added vertical padding */
  border-bottom: 1px solid ${productionColors.border.secondary};
  margin-bottom: ${enhancedSpacing[3]};
`;

const WeatherIcon = styled.div<{ condition?: string | null }>`
  width: 60px;
  height: 60px;
  border-radius: 12px;
  background: ${props => getWeatherIconBackground(props.condition)};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
`;

const TemperatureDisplay = styled.div`
  flex: 1;
`;

const Temperature = styled.div`
  font-size: 2.25rem; /* Slightly smaller for density */
  font-weight: 700;
  color: ${productionColors.text.primary};
  line-height: 1;
  display: flex;
  align-items: baseline;
  gap: 4px;
  
  .unit {
    font-size: 1rem;
    font-weight: 400;
    color: ${productionColors.text.secondary};
  }
`;

const FeelsLike = styled.div`
  font-size: 0.8rem; /* Slightly smaller */
  color: ${productionColors.text.tertiary};
  margin-top: 2px; /* Reduced margin
`;

const WeatherDescription = styled.div`
  font-size: 1rem;
  color: #D1D5DB;
  text-transform: capitalize;
  margin-top: 8px;
`;

const WeatherMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${enhancedSpacing[3]}; /* Tighter gaps for density */
  margin-top: ${enhancedSpacing[3]};
  
  @media (max-width: 480px) {
    grid-template-columns: 1fr; /* Single column on small screens */
  }
`;

const MetricItem = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background: linear-gradient(135deg, ${productionColors.background.tertiary}, ${productionColors.background.secondary});
  border-radius: 8px;
  border: 1px solid ${productionColors.border.secondary};
  transition: all ${productionAnimations.duration.normal} ${productionAnimations.easing.smooth};
  position: relative;
  overflow: hidden;
  
  /* Subtle shine effect */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
    transition: left 0.5s ease;
  }
  
  &:hover {
    background: linear-gradient(135deg, ${productionColors.interactive.hover}, ${productionColors.background.tertiary});
    border-color: ${productionColors.border.accent};
    transform: translateY(-2px);
    box-shadow: 
      0 4px 12px rgba(0, 0, 0, 0.15),
      0 0 0 1px rgba(239, 68, 68, 0.1);
    
    &::before {
      left: 100%;
    }
  }
`;

const MetricIcon = styled.div`
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
  display: flex;
  align-items: center;
  justify-content: center;
  color: #EF4444;
  box-shadow: 
    0 2px 8px rgba(239, 68, 68, 0.2),
    inset 0 1px 1px rgba(255, 255, 255, 0.1);
  transition: all ${productionAnimations.duration.normal} ${productionAnimations.easing.smooth};
  
  ${MetricItem}:hover & {
    transform: scale(1.1) rotate(5deg);
    box-shadow: 
      0 4px 12px rgba(239, 68, 68, 0.3),
      inset 0 1px 1px rgba(255, 255, 255, 0.15);
  }
`;

const MetricDetails = styled.div`
  flex: 1;
`;

const MetricValue = styled.div`
  font-size: 0.875rem;
  font-weight: 600;
  color: #F7F8FA;
`;

const MetricLabel = styled.div`
  font-size: 0.75rem;
  color: #9CA3AF;
`;

const DangerAssessment = styled.div`
  ${productionCard}
  display: flex;
  flex-direction: column;
  gap: ${enhancedSpacing[3]};
  min-height: 320px;
  position: relative;
  overflow: hidden;
  
  /* Animated gradient top bar */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #EF4444, #F59E0B, #22C55E);
    z-index: 1;
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
  }
  
  /* Subtle radial glow */
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(239, 68, 68, 0.08), transparent 70%);
    pointer-events: none;
    z-index: 0;
  }
  
  /* Content above glow */
  & > * {
    position: relative;
    z-index: 1;
  }
`;

const DangerHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const DangerLevel = styled.div<{ level: 'low' | 'medium' | 'high' | 'critical' }>`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  border-radius: 10px;
  font-size: 0.875rem;
  font-weight: 600;
  background: ${props => getDangerLevelBackground(props.level)};
  color: ${props => getDangerLevelColor(props.level)};
  border: 1px solid ${props => getDangerLevelBorder(props.level)};
  box-shadow: 0 4px 12px ${props => {
    const level = props.level;
    if (level === 'critical') return 'rgba(239, 68, 68, 0.3)';
    if (level === 'high') return 'rgba(245, 158, 11, 0.3)';
    if (level === 'medium') return 'rgba(59, 130, 246, 0.2)';
    return 'rgba(34, 197, 94, 0.2)';
  }};
  transition: all ${productionAnimations.duration.normal} ${productionAnimations.easing.smooth};
  
  &:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 16px ${props => {
      const level = props.level;
      if (level === 'critical') return 'rgba(239, 68, 68, 0.4)';
      if (level === 'high') return 'rgba(245, 158, 11, 0.4)';
      if (level === 'medium') return 'rgba(59, 130, 246, 0.3)';
      return 'rgba(34, 197, 94, 0.3)';
    }};
  }
`;

const AlertsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const AlertItem = styled.div<{ severity: 'info' | 'warning' | 'critical' }>`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 10px;
  background: ${props => getAlertBackground(props.severity)};
  border: 1px solid ${props => getAlertBorder(props.severity)};
  transition: all ${productionAnimations.duration.normal} ${productionAnimations.easing.smooth};
  position: relative;
  overflow: hidden;
  
  /* Severity indicator bar */
  &::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: ${props => {
      if (props.severity === 'critical') return '#EF4444';
      if (props.severity === 'warning') return '#F59E0B';
      return '#3B82F6';
    }};
    box-shadow: 0 0 8px ${props => {
      if (props.severity === 'critical') return 'rgba(239, 68, 68, 0.5)';
      if (props.severity === 'warning') return 'rgba(245, 158, 11, 0.5)';
      return 'rgba(59, 130, 246, 0.5)';
    }};
  }
  
  &:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 12px ${props => {
      if (props.severity === 'critical') return 'rgba(239, 68, 68, 0.2)';
      if (props.severity === 'warning') return 'rgba(245, 158, 11, 0.2)';
      return 'rgba(59, 130, 246, 0.15)';
    }};
  }
`;

// Enhanced status section for better information density
const StatusSection = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${enhancedSpacing[2]} 0;
  border-top: 1px solid ${productionColors.border.secondary};
  margin-top: auto;
`;

const StatusItem = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: ${productionColors.text.tertiary};
  
  .status-icon {
    width: 12px;
    height: 12px;
  }
`;

const LastUpdated = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  color: ${productionColors.text.tertiary};
`;

function getWeatherIconBackground(condition?: string | null): string {
  if (!condition || typeof condition !== 'string') {
    return 'linear-gradient(135deg, #6B7280 0%, #4B5563 100%)';
  }
  switch (condition.toLowerCase()) {
    case 'clear':
    case 'sunny':
      return 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)';
    case 'clouds':
    case 'cloudy':
      return 'linear-gradient(135deg, #6B7280 0%, #4B5563 100%)';
    case 'rain':
    case 'drizzle':
      return 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)';
    case 'snow':
      return 'linear-gradient(135deg, #E5E7EB 0%, #9CA3AF 100%)';
    case 'thunderstorm':
      return 'linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%)';
    default:
      return 'linear-gradient(135deg, #6B7280 0%, #4B5563 100%)';
  }
}

function getWeatherIcon(condition: string | undefined | null) {
  if (!condition || typeof condition !== 'string') {
    return <Cloud size={28} />;
  }
  switch (condition.toLowerCase()) {
    case 'clear':
    case 'sunny':
      return <Sun size={28} />;
    case 'clouds':
    case 'cloudy':
      return <Cloud size={28} />;
    case 'rain':
    case 'drizzle':
      return <CloudRain size={28} />;
    case 'snow':
      return <CloudSnow size={28} />;
    case 'thunderstorm':
      return <Zap size={28} />;
    default:
      return <Cloud size={28} />;
  }
}

function getDangerLevelBackground(level: string): string {
  switch (level) {
    case 'low':
      return 'rgba(34, 197, 94, 0.1)';
    case 'medium':
      return 'rgba(251, 191, 36, 0.1)';
    case 'high':
      return 'rgba(239, 68, 68, 0.1)';
    case 'critical':
      return 'rgba(147, 51, 234, 0.1)';
    default:
      return 'rgba(107, 114, 128, 0.1)';
  }
}

function getDangerLevelColor(level: string): string {
  switch (level) {
    case 'low':
      return '#22C55E';
    case 'medium':
      return '#FBBF24';
    case 'high':
      return '#EF4444';
    case 'critical':
      return '#9333EA';
    default:
      return '#9CA3AF';
  }
}

function getDangerLevelBorder(level: string): string {
  switch (level) {
    case 'low':
      return 'rgba(34, 197, 94, 0.2)';
    case 'medium':
      return 'rgba(251, 191, 36, 0.2)';
    case 'high':
      return 'rgba(239, 68, 68, 0.2)';
    case 'critical':
      return 'rgba(147, 51, 234, 0.2)';
    default:
      return 'rgba(107, 114, 128, 0.2)';
  }
}

function getAlertBackground(severity: string): string {
  switch (severity) {
    case 'info':
      return 'rgba(59, 130, 246, 0.1)';
    case 'warning':
      return 'rgba(251, 191, 36, 0.1)';
    case 'critical':
      return 'rgba(239, 68, 68, 0.1)';
    default:
      return 'rgba(107, 114, 128, 0.1)';
  }
}

function getAlertBorder(severity: string): string {
  switch (severity) {
    case 'info':
      return 'rgba(59, 130, 246, 0.2)';
    case 'warning':
      return 'rgba(251, 191, 36, 0.2)';
    case 'critical':
      return 'rgba(239, 68, 68, 0.2)';
    default:
      return 'rgba(107, 114, 128, 0.2)';
  }
}

function calculateDangerLevel(weatherData: EnhancedWeatherData): 'low' | 'medium' | 'high' | 'critical' {
  if (!weatherData) return 'low';
  
  const { temperature, humidity, windSpeed, pressure } = weatherData;
  let riskScore = 0;
  
  // Temperature risks
  if (temperature > 35 || temperature < -10) riskScore += 2;
  else if (temperature > 30 || temperature < 0) riskScore += 1;
  
  // Wind risks
  if (windSpeed > 25) riskScore += 2;
  else if (windSpeed > 15) riskScore += 1;
  
  // Humidity risks
  if (humidity > 90 || humidity < 20) riskScore += 1;
  
  // Pressure risks (indicating storms)
  if (pressure < 980) riskScore += 2;
  else if (pressure < 1000) riskScore += 1;
  
  if (riskScore >= 5) return 'critical';
  if (riskScore >= 3) return 'high';
  if (riskScore >= 1) return 'medium';
  return 'low';
}

const WeatherDashboardPanel: React.FC = () => {
  const [weatherData, setWeatherData] = useState<EnhancedWeatherData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [currentLocation, setCurrentLocation] = useState<{ lat: number, lon: number, city: string } | null>(null);

  // Fetch current location first - defaults to Jaipur
  useEffect(() => {
    const getCurrentLocation = async () => {
      try {
        console.log('📍 Fetching location for weather (default: Jaipur, India)...');
        const location = await enhancedLocationService.getCurrentLocation(true);
        console.log('✅ Got location:', location);
        setCurrentLocation({
          lat: location.latitude,
          lon: location.longitude,
          city: `${location.city}, ${location.state}`
        });
      } catch (error) {
        console.error('❌ Failed to get location, using Jaipur as default:', error);
        // Always fallback to Jaipur, Rajasthan coordinates
        setCurrentLocation({
          lat: 26.9124,
          lon: 75.7873,
          city: 'Jaipur, Rajasthan'
        });
      }
    };

    getCurrentLocation();
  }, []);

  useEffect(() => {
    if (!currentLocation) return;

    const fetchWeatherData = async () => {
      try {
        setLoading(true);
        console.log(`🌤️ Fetching weather for: ${currentLocation.city} (${currentLocation.lat}, ${currentLocation.lon})`);
        const response = await CachedAPIService.getEnhancedWeatherData(currentLocation.lat, currentLocation.lon);
        if (response.data) {
          console.log('✅ Weather data received:', response.data);
          setWeatherData(response.data);
          setLastUpdate(new Date());
        }
      } catch (error) {
        console.error('Failed to fetch weather data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchWeatherData();
    
    // Update every 10 minutes
    const interval = setInterval(fetchWeatherData, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [currentLocation]);

  if (loading) {
    return (
      <WeatherContainer>
        <CurrentWeatherCard>
          <div style={{ textAlign: 'center', color: '#9CA3AF' }}>
            Loading weather data...
          </div>
        </CurrentWeatherCard>
      </WeatherContainer>
    );
  }

  if (!weatherData) {
    return (
      <WeatherContainer>
        <CurrentWeatherCard>
          <div style={{ textAlign: 'center', color: '#EF4444' }}>
            Unable to load weather data
          </div>
        </CurrentWeatherCard>
      </WeatherContainer>
    );
  }

  const dangerLevel = calculateDangerLevel(weatherData);
  
  return (
    <WeatherContainer>
      <CurrentWeatherCard>
        <WeatherHeader>
          <WeatherTitle>
            <Activity size={20} />
            Current Weather
          </WeatherTitle>
          <LocationBadge>
            <MapPin size={12} />
            {currentLocation?.city || weatherData.location?.city || 'Unknown Location'}
          </LocationBadge>
        </WeatherHeader>

        <CurrentConditions>
          <WeatherIcon condition={weatherData.condition}>
            {getWeatherIcon(weatherData.condition)}
          </WeatherIcon>
          
          <TemperatureDisplay>
            <Temperature>
              {Math.round(weatherData.temperature)}
              <span className="unit">°C</span>
            </Temperature>
            <FeelsLike>Feels like {Math.round(weatherData.feelsLike || weatherData.temperature)}°C</FeelsLike>
            <WeatherDescription>{weatherData.description}</WeatherDescription>
          </TemperatureDisplay>
        </CurrentConditions>

        <WeatherMetrics>
          <MetricItem>
            <MetricIcon>
              <Droplets size={16} />
            </MetricIcon>
            <MetricDetails>
              <MetricValue>{weatherData.humidity}%</MetricValue>
              <MetricLabel>Humidity</MetricLabel>
            </MetricDetails>
          </MetricItem>

          <MetricItem>
            <MetricIcon>
              <Wind size={16} />
            </MetricIcon>
            <MetricDetails>
              <MetricValue>{weatherData.windSpeed} km/h</MetricValue>
              <MetricLabel>Wind Speed</MetricLabel>
            </MetricDetails>
          </MetricItem>

          <MetricItem>
            <MetricIcon>
              <Gauge size={16} />
            </MetricIcon>
            <MetricDetails>
              <MetricValue>{weatherData.pressure} hPa</MetricValue>
              <MetricLabel>Pressure</MetricLabel>
            </MetricDetails>
          </MetricItem>

          <MetricItem>
            <MetricIcon>
              <Eye size={16} />
            </MetricIcon>
            <MetricDetails>
              <MetricValue>{weatherData.visibility} km</MetricValue>
              <MetricLabel>Visibility</MetricLabel>
            </MetricDetails>
          </MetricItem>
        </WeatherMetrics>

        <StatusSection>
          <StatusItem>
            <Activity className="status-icon" />
            Live
          </StatusItem>
          <StatusItem>
            <MapPin className="status-icon" />
            GPS
          </StatusItem>
          <LastUpdated>
            <Clock size={12} />
            {lastUpdate.toLocaleTimeString([], { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </LastUpdated>
        </StatusSection>
      </CurrentWeatherCard>

      <DangerAssessment>
        <DangerHeader>
          <WeatherTitle>
            <AlertTriangle size={20} />
            Weather Risk Assessment
          </WeatherTitle>
        </DangerHeader>

        <DangerLevel level={dangerLevel}>
          {dangerLevel === 'low' ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
          {dangerLevel.charAt(0).toUpperCase() + dangerLevel.slice(1)} Risk
        </DangerLevel>

        <AlertsList>
          {weatherData.temperature > 35 && (
            <AlertItem severity="critical">
              <Thermometer size={16} style={{ color: '#EF4444' }} />
              <div>
                <div style={{ color: '#F7F8FA', fontWeight: 600 }}>Extreme Heat Warning</div>
                <div style={{ color: '#9CA3AF', fontSize: '0.75rem' }}>
                  Temperature exceeds safe outdoor activity levels
                </div>
              </div>
            </AlertItem>
          )}

          {weatherData.windSpeed > 25 && (
            <AlertItem severity="warning">
              <Wind size={16} style={{ color: '#FBBF24' }} />
              <div>
                <div style={{ color: '#F7F8FA', fontWeight: 600 }}>High Wind Advisory</div>
                <div style={{ color: '#9CA3AF', fontSize: '0.75rem' }}>
                  Strong winds may affect outdoor operations
                </div>
              </div>
            </AlertItem>
          )}

          {weatherData.pressure < 980 && (
            <AlertItem severity="critical">
              <Gauge size={16} style={{ color: '#EF4444' }} />
              <div>
                <div style={{ color: '#F7F8FA', fontWeight: 600 }}>Severe Weather Alert</div>
                <div style={{ color: '#9CA3AF', fontSize: '0.75rem' }}>
                  Low pressure indicates potential storms
                </div>
              </div>
            </AlertItem>
          )}

          {dangerLevel === 'low' && (
            <AlertItem severity="info">
              <CheckCircle size={16} style={{ color: '#22C55E' }} />
              <div>
                <div style={{ color: '#F7F8FA', fontWeight: 600 }}>Favorable Conditions</div>
                <div style={{ color: '#9CA3AF', fontSize: '0.75rem' }}>
                  Weather conditions are safe for normal operations
                </div>
              </div>
            </AlertItem>
          )}
        </AlertsList>

        <LastUpdated>
          <Activity size={12} />
          Risk assessment updated {lastUpdate.toLocaleTimeString()}
        </LastUpdated>
      </DangerAssessment>
    </WeatherContainer>
  );
};

export { WeatherDashboardPanel };
export default WeatherDashboardPanel;