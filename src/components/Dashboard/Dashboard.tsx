import React, { useCallback, useState, useEffect, Suspense } from 'react';
import styled, { keyframes } from 'styled-components';
import { spacing, breakpoints } from '../../styles/spacing';
import { enhancedSpacing, enhancedGrid } from '../../styles/enhanced-design-system';
import { 
  productionColors, 
  productionAnimations, 
  productionCard,
  productionScrollbar 
} from '../../styles/production-ui-system';
import CurrentAlerts from './CurrentAlerts';
import MLPredictionAccuracy from './MLPredictionAccuracy';
import ActionButtons from './ActionButtons';
import SevenDayForecast from './SevenDayForecast';
import HistoricalTrends from './HistoricalTrends';
import GeolocationManager from '../Location/GeolocationManager';
import EmergencyResponsePanel from '../Emergency/EmergencyResponsePanel';
import EvacuationSafetyModule from '../Safety/EvacuationSafetyModule';
import ResourceManagementDashboard from '../Resources/ResourceManagementDashboard';
import CommunicationHub from '../Communication/CommunicationHub';
import EnhancedWeatherWidget from '../Dashboard/EnhancedWeatherWidget';
import AirQualityWidget from './AirQualityWidget';
import { SystemDiagnostics } from '../Diagnostics/SystemDiagnostics';
import { LoadingOverlay, SkeletonDashboard } from '../Layout/LoadingStates';
import { useAutoRefresh, useRefreshSettings } from '../../hooks/useAutoRefresh';
import { useLiveDataExport } from '../../services/liveDataExport';
import { RefreshCw, Clock, Download } from 'lucide-react';
import { enhancedLocationService } from '../../services/enhancedLocationService';
import SimpleWeatherService from '../../services/simpleWeatherService';
import WeatherForecastService from '../../services/weatherForecastService';
import airQualityService, { AQIData } from '../../services/airQualityService';
import RiskCalculationService, { WeatherRiskFactors } from '../../services/riskCalculationService';
import logger from '../../utils/logger';
import GlobeRiskHero from './GlobeRiskHero';
import { ForecastData } from '../../types';

// Alert risk severity weights for risk calculation
const SEVERITY_WEIGHTS = {
  critical: 10,
  high: 7,
  moderate: 5,
  low: 2,
  info: 1,
} as const;

/**
 * Calculate aggregated risk score from active alerts
 * @param alerts Array of current alerts
 * @returns Normalized risk score 0-10
 */
const calculateAlertRisk = (alerts: any[]): number => {
  if (!alerts || alerts.length === 0) return 0;
  
  const totalWeight = alerts.reduce((sum, alert) => {
    const severity = alert.severity?.toLowerCase() || 'low';
    const weight = SEVERITY_WEIGHTS[severity as keyof typeof SEVERITY_WEIGHTS] || 1;
    return sum + weight;
  }, 0);
  
  // Normalize to 0-10 scale based on average severity
  // Max possible average is 10 (all critical), min is 1 (all info)
  const averageWeight = totalWeight / alerts.length;
  const normalizedRisk = Math.min(averageWeight, 10);
  
  logger.log(`📊 Alert risk calculated: ${alerts.length} alerts, risk: ${normalizedRisk.toFixed(2)}`);
  
  return normalizedRisk;
};

const DashboardContainer = styled.main`
  min-height: 100vh;
  padding-top: 64px; /* Updated for 64px navigation bar */
  background: transparent; /* Let starfield show through */
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  color: ${productionColors.text.primary};
  
  /* Enhanced smooth scrolling */
  ${productionScrollbar}
  
  ${productionAnimations.keyframes.spin}
`;

// ENHANCED 16/24px GRID SYSTEM - Zero overlapping guarantee
const DashboardGrid = styled.div`
  display: grid;
  gap: ${enhancedGrid.containerGap}; /* 24px perfect gaps */
  padding: ${enhancedGrid.containerPadding}; /* 24px container padding */
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  grid-auto-rows: minmax(0, auto); /* Prevent children from overflowing their grid cells */
  
  /* Desktop: 3-column layout (1200px+) */
  @media (min-width: ${breakpoints.desktop}) {
    grid-template-columns: 380px 1fr 380px; /* Fixed 380px sidebars, flexible center */
    grid-template-areas:
      "left center right"
      "weather weather weather"
      "diagnostics diagnostics diagnostics"
      "emergency emergency emergency"
      "evacuation evacuation evacuation"
      "resources resources resources"
      "communication communication communication";
    gap: 24px; /* 24px gap as requested */
  }
  
  /* Tablet: 2-column layout (768px-1199px) */
  @media (min-width: ${breakpoints.tablet}) and (max-width: ${breakpoints.tabletMax}) {
    grid-template-columns: 1fr 1fr;
    grid-template-areas:
      "left center"
      "right center"
      "weather weather"
      "diagnostics diagnostics"
      "emergency emergency"
      "evacuation resources"
      "communication communication";
    gap: ${enhancedGrid.cardGap}; /* 16px on tablet - enhanced grid compliant */
    padding: ${enhancedGrid.minMargin}; /* 16px padding on tablet */
  }
  
  /* Mobile: Single column (below 768px) */
  @media (max-width: ${breakpoints.mobile}) {
    grid-template-columns: 1fr;
    grid-template-areas:
      "center"
      "weather"
      "diagnostics"
      "left"
      "right"
      "emergency"
      "evacuation"
      "resources"
      "communication";
    gap: ${enhancedGrid.cardGap}; /* 16px on mobile - enhanced grid compliant */
    padding: ${enhancedGrid.minMargin}; /* 16px on mobile - enhanced grid compliant */
  }
`;

const LeftSidebar = styled.aside`
  grid-area: left;
  display: flex;
  flex-direction: column;
  gap: ${enhancedGrid.cardGap}; /* 16px between cards - enhanced grid compliant */
  min-height: 0; /* Prevent flex overflow */
`;

const CenterArea = styled.section`
  grid-area: center;
  display: flex;
  flex-direction: column;
  gap: ${enhancedGrid.cardGap}; /* 16px between elements - enhanced grid compliant */
  align-items: center;
  justify-content: flex-start;
  min-height: 0;
  width: 100%;
`;

const RightSidebar = styled.aside`
  grid-area: right;
  display: flex;
  flex-direction: column;
  gap: ${enhancedGrid.cardGap}; /* 16px between cards - enhanced grid compliant */
  min-height: 0;
`;

// Remove the old VisualizationContainer as it's now integrated into GlobeRiskHero

// PRODUCTION CARD STYLING - Enhanced with cinematic animations
const fadeInUp = keyframes`
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

const DashboardCard = styled.div<{ animationDelay?: number }>`
  ${productionCard}
  
  /* Enhanced professional spacing */
  padding: ${enhancedSpacing[6]}; /* 24px internal padding */
  box-sizing: border-box;
  width: 100%;
  gap: ${enhancedSpacing[4]}; /* 16px internal element gaps */
  
  /* PRODUCTION POLISH: Prominent glowing shadow on pure black */
  box-shadow: 0 8px 36px rgba(0, 0, 0, 0.7);
  background: rgba(22, 24, 29, 0.95); /* Glassmorphism for pure black */
  backdrop-filter: blur(10px);
  
  /* Production animation system with stagger effect */
  animation: ${fadeInUp} ${productionAnimations.duration.slower} ${productionAnimations.easing.smooth};
  animation-delay: ${({ animationDelay }) => animationDelay || 0}ms;
  animation-fill-mode: both;
  
  /* ENHANCED HOVER: Stronger glow and lift */
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  
  &:hover {
    border-color: ${productionColors.border.accent};
    box-shadow: 
      0 12px 48px rgba(0, 0, 0, 0.52),
      0 6px 24px rgba(0, 0, 0, 0.28),
      inset 0 1px 0 rgba(255, 255, 255, 0.1),
      0 0 0 1px rgba(239, 68, 68, 0.15);
    transform: translateY(-6px);
  }
  
  /* Zero overlap guarantee */
  position: relative;
  z-index: 1;
  overflow: auto;
  
  /* Production text color */
  color: ${productionColors.text.primary};
  
  /* Subtle inner glow - enhanced */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(239, 68, 68, 0.15), transparent);
    pointer-events: none;
  }
`;

// Emergency section spans full width
const EmergencySection = styled.section`
  grid-area: emergency;
  display: grid;
  grid-template-columns: 1fr;
  gap: ${spacing.lg}; /* 16px gap */
`;

const EvacuationSection = styled.section`
  grid-area: evacuation;
  display: grid;
  grid-template-columns: 1fr;
  gap: ${spacing.lg};
`;

const ResourcesSection = styled.section`
  grid-area: resources;
  display: grid;
  grid-template-columns: 1fr;
  gap: ${spacing.lg};
`;

const CommunicationSection = styled.section`
  grid-area: communication;
  display: grid;
  grid-template-columns: 1fr;
  gap: ${spacing.lg};
`;

const WeatherSection = styled.section`
  grid-area: weather;
`;

const DiagnosticsSection = styled.section`
  grid-area: diagnostics;
`;

// Live Data Status Bar
const LiveDataStatusBar = styled.div`
  position: fixed;
  top: 70px;
  left: 0;
  right: 0;
  height: 32px;
  background: ${({ theme }) => theme.colors.surface.elevated};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 ${enhancedSpacing[6]};
  z-index: 1000;
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.secondary};
  
  .status-section {
    display: flex;
    align-items: center;
    gap: ${enhancedSpacing[3]};
  }
  
  .refresh-indicator {
    display: flex;
    align-items: center;
    gap: ${enhancedSpacing[1]};
    color: ${({ theme }) => theme.colors.success};
    
    &.refreshing {
      color: ${({ theme }) => theme.colors.primary[400]};
    }
    
    svg {
      width: 12px;
      height: 12px;
    }
  }
  
  .next-refresh {
    font-size: 10px;
    opacity: 0.8;
  }
`;

const Dashboard: React.FC = () => {
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [dashboardLoaded, setDashboardLoaded] = useState(false);
  const [globalRiskScore, setGlobalRiskScore] = useState<number>(0);
  const [isCalculatingRisk, setIsCalculatingRisk] = useState(true);
  const [forecastData, setForecastData] = useState<ForecastData[] | null>(null);
  const [aqiData, setAqiData] = useState<AQIData | null>(null);
  const [aqiLoading, setAqiLoading] = useState(false);
  
  const { refreshInterval, autoRefreshEnabled } = useRefreshSettings();
  const { lastRefresh, nextRefresh, isRefreshing, manualRefresh } = useAutoRefresh({
    interval: Math.max(refreshInterval, 5), // Minimum 5 minutes to prevent excessive API calls
    enabled: autoRefreshEnabled,
    onRefresh: () => {
      logger.log('🔄 Dashboard data refreshed via auto-refresh');
      calculateGlobalRisk(); // Recalculate risk on refresh
    }
  });
  
  const { exportBothFormats, hasData } = useLiveDataExport();

  // Fetch 7-day forecast data
  const fetchForecastData = useCallback(async () => {
    try {
      const loc = await enhancedLocationService.getCurrentLocation();
      const forecast = await WeatherForecastService.getForecast(loc.latitude, loc.longitude, 7);
      
      // Convert to component format
      const convertedForecast = WeatherForecastService.convertToForecastData(forecast.forecast);
      setForecastData(convertedForecast);
      
      logger.log('📅 7-day forecast loaded:', {
        days: convertedForecast.length,
        source: forecast.source,
        is_real: forecast.is_real
      });
    } catch (error) {
      logger.error('❌ Forecast fetch failed:', error);
      setForecastData(null);
    }
  }, []);

  // Fetch air quality data
  const fetchAQIData = useCallback(async () => {
    try {
      setAqiLoading(true);
      const loc = await enhancedLocationService.getCurrentLocation();
      const aqi = await airQualityService.getAirQuality(loc.latitude, loc.longitude);
      
      setAqiData(aqi);
      
      logger.log('🌬️ Air quality data loaded:', {
        aqi: aqi.aqi,
        level: aqi.level,
        description: aqi.description,
        shouldAlert: airQualityService.shouldAlert(aqi.aqi)
      });
    } catch (error) {
      logger.error('❌ AQI fetch failed:', error);
      setAqiData(null);
    } finally {
      setAqiLoading(false);
    }
  }, []);

  // Calculate global risk score from live data
  const calculateGlobalRisk = useCallback(async () => {
    try {
      setIsCalculatingRisk(true);
      
      // Fetch live weather data using simple reliable service
      const loc = await enhancedLocationService.getCurrentLocation();
      const weatherData = await SimpleWeatherService.getWeather(loc.latitude, loc.longitude);
      
      const weatherFactors: WeatherRiskFactors = {
        temp: weatherData.current.temp,
        feelsLike: weatherData.current.feels_like,
        condition: weatherData.current.weather[0].main,
        humidity: weatherData.current.humidity,
        windSpeed: Math.round(weatherData.current.wind_speed * 3.6), // m/s to km/h
        pressure: weatherData.current.pressure,
        visibility: weatherData.current.visibility,
        uvIndex: weatherData.current.uvi
      };
      
      // Calculate weather risk
      const weatherRisk = RiskCalculationService.calculateWeatherRisk(weatherFactors);
      
      // Calculate alert risk from current alerts
      // TODO: Implement actual alerts state management
      const alertRisk = calculateAlertRisk([]);
      
      // Get pollution risk factor (0-1) and convert to 0-10 scale
      const pollutionRisk = aqiData 
        ? airQualityService.getPollutionRiskFactor(aqiData.aqi) * 10
        : 0;
      
      // Calculate global risk with pollution factor
      const baseGlobalRisk = RiskCalculationService.calculateGlobalRisk(weatherRisk, alertRisk);
      const globalRisk = Math.min(10, baseGlobalRisk + (pollutionRisk * 0.3)); // Pollution adds up to 30% boost
      
      setGlobalRiskScore(globalRisk);
      logger.log('🎯 Global risk calculated:', {
        weather: weatherRisk,
        alerts: alertRisk,
        pollution: pollutionRisk,
        global: globalRisk,
        location: `${loc.city}, ${loc.state || loc.country}`
      });
    } catch (error) {
      logger.error('❌ Risk calculation failed:', error);
      setGlobalRiskScore(0); // Default to low risk on error
    } finally {
      setIsCalculatingRisk(false);
    }
  }, [aqiData]);

  // Load dashboard data on mount and set up refresh interval
  useEffect(() => {
    const loadDashboard = async () => {
      // Fetch all data in parallel
      await Promise.all([
        fetchForecastData(),
        fetchAQIData(),
        calculateGlobalRisk()
      ]);
      
      // Simulate component initialization
      await new Promise(resolve => setTimeout(resolve, 1500));
      setDashboardLoaded(true);
      
      // Remove loading overlay after smooth transition
      setTimeout(() => {
        setIsInitialLoading(false);
      }, 300);
    };

    loadDashboard();
    
    // Set up periodic refresh every 5 minutes
    const refreshInterval = setInterval(() => {
      fetchForecastData();
      fetchAQIData();
      calculateGlobalRisk();
    }, 5 * 60 * 1000); // 5 minutes
    
    return () => clearInterval(refreshInterval);
  }, []); // Run only once on mount

  // Recalculate risk when AQI data changes (but don't fetch AQI again)
  useEffect(() => {
    if (aqiData && dashboardLoaded) {
      calculateGlobalRisk();
    }
  }, [aqiData, dashboardLoaded, calculateGlobalRisk]);

  // Memoize time formatting to prevent excessive re-renders
  const formatTimeUntilNextRefresh = useCallback((nextRefresh: Date | null) => {
    if (!nextRefresh) return '';
    const now = new Date();
    const diff = nextRefresh.getTime() - now.getTime();
    if (diff <= 0) return '0:00';
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, []);

  const handleDownloadReport = useCallback(() => {
    logger.log('📄 Downloading comprehensive live data report...');
    exportBothFormats(); // Export both PDF and CSV
  }, [exportBothFormats]);

  const handleRefreshData = useCallback(() => {
    logger.log('🔄 Manual dashboard refresh triggered...');
    manualRefresh();
  }, [manualRefresh]);

  // Show loading screen during initial load
  if (isInitialLoading) {
    return (
      <DashboardContainer>
        <LoadingOverlay 
          message="Initializing Alert Aid Dashboard..." 
          fullScreen={false}
        />
        {!dashboardLoaded && <SkeletonDashboard />}
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      {/* Live Data Status Bar */}
      <LiveDataStatusBar>
        <div className="status-section">
          <div className={`refresh-indicator ${isRefreshing ? 'refreshing' : ''}`}>
            <RefreshCw size={12} style={{ 
              animation: isRefreshing ? 'spin 1s linear infinite' : 'none' 
            }} />
            <span>
              {isRefreshing ? 'Refreshing...' : 
               autoRefreshEnabled ? `Auto-refresh: ${refreshInterval}min` : 'Auto-refresh: Off'}
            </span>
          </div>
          
          {autoRefreshEnabled && nextRefresh && (
            <div className="next-refresh">
              <Clock size={10} />
              <span>Next: {formatTimeUntilNextRefresh(nextRefresh)}</span>
            </div>
          )}
        </div>
        
        <div className="status-section">
          <span>Last updated: {lastRefresh.toLocaleTimeString()}</span>
          {hasData && (
            <button 
              onClick={handleDownloadReport}
              style={{
                background: 'none',
                border: 'none', 
                color: 'inherit',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '10px'
              }}
            >
              <Download size={10} />
              Export Data
            </button>
          )}
        </div>
      </LiveDataStatusBar>

      <GeolocationManager />
      
      <DashboardGrid className="dashboard-grid" style={{ paddingTop: '32px' }}> {/* Account for status bar */}
        {/* Left Sidebar - Alerts & Controls */}
        <LeftSidebar>
          <DashboardCard animationDelay={100}>
            <CurrentAlerts />
          </DashboardCard>
          
          <DashboardCard animationDelay={200}>
            <MLPredictionAccuracy />
          </DashboardCard>
          
          <DashboardCard animationDelay={300}>
            <ActionButtons 
              onDownloadReport={handleDownloadReport}
              onRefreshData={handleRefreshData}
            />
          </DashboardCard>
        </LeftSidebar>

        {/* Center Area - Merged Globe + Risk Hero */}
        <CenterArea>
          <Suspense fallback={<LoadingOverlay message="Loading Globe..." />}>
            <GlobeRiskHero 
              score={globalRiskScore} 
              isCalculating={isCalculatingRisk}
              alerts={[]}
            />
          </Suspense>
        </CenterArea>

        {/* Right Sidebar - Forecast & Trends */}
        <RightSidebar>
          <DashboardCard animationDelay={250}>
            <SevenDayForecast forecast={forecastData || undefined} />
          </DashboardCard>
          
          <DashboardCard animationDelay={350}>
            <HistoricalTrends />
          </DashboardCard>
        </RightSidebar>

        {/* Weather Dashboard - Full Width with Enhanced Widget */}
        <WeatherSection>
          <EnhancedWeatherWidget />
          <AirQualityWidget aqiData={aqiData} loading={aqiLoading} />
        </WeatherSection>

        {/* System Diagnostics - Full Width */}
        <DiagnosticsSection>
          <DashboardCard animationDelay={375}>
            <SystemDiagnostics />
          </DashboardCard>
        </DiagnosticsSection>

        {/* Emergency Response - Full Width */}
        <EmergencySection>
          <DashboardCard animationDelay={400}>
            <EmergencyResponsePanel />
          </DashboardCard>
        </EmergencySection>

        {/* Evacuation & Safety */}
        <EvacuationSection>
          <DashboardCard animationDelay={450}>
            <EvacuationSafetyModule />
          </DashboardCard>
        </EvacuationSection>

        {/* Resource Management */}
        <ResourcesSection>
          <DashboardCard animationDelay={500}>
            <ResourceManagementDashboard />
          </DashboardCard>
        </ResourcesSection>

        {/* Communication Hub */}
        <CommunicationSection>
          <DashboardCard animationDelay={550}>
            <CommunicationHub />
          </DashboardCard>
        </CommunicationSection>
      </DashboardGrid>
    </DashboardContainer>
  );
};

export default Dashboard;