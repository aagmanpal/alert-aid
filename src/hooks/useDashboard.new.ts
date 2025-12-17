import { useDisasterData, useApiHealth } from './useDisasterData';
import { useGeolocation } from '../components/Location/GeolocationManager';

// Main dashboard hook that combines all disaster management data
export function useDashboard() {
  const { location } = useGeolocation();
  const { isHealthy, checkHealth } = useApiHealth();
  const {
    riskPrediction,
    weather,
    alerts,
    forecast,
    modelPerformance,
    loading,
    errors,
    lastUpdated,
    refreshAllData,
    clearError,
  } = useDisasterData(location);

  return {
    // Location data
    location,
    
    // API health
    isApiHealthy: isHealthy,
    checkApiHealth: checkHealth,
    
    // Disaster data
    riskPrediction,
    weather,
    alerts: alerts?.alerts || [],
    alertsCount: alerts?.count || 0,
    forecast: [], // Endpoint removed
    modelPerformance,
    
    // Loading states
    isLoading: Object.values(loading).some(Boolean),
    loadingStates: loading,
    
    // Error states  
    hasErrors: Object.values(errors).some(error => error !== null),
    errors,
    
    // Metadata
    lastUpdated,
    
    // Actions
    refreshData: refreshAllData,
    clearError,
    
    // Computed properties for dashboard display
    overallRisk: riskPrediction?.overall_risk || 'unknown',
  riskLevel: riskPrediction?.overall_risk || 'unknown',
    activeAlertsCount: alerts?.alerts?.filter(alert => 
      alert.severity === 'High' || alert.urgency === 'Immediate'
    ).length || 0,
    weatherSummary: weather ? {
      temp: Math.round(weather.temperature),
      condition: weather.conditions,
      humidity: weather.humidity,
      wind: weather.wind_speed,
    } : null,
  };
}

// Helper function to determine risk level
function getRiskLevel(risk: number): 'low' | 'medium' | 'high' | 'critical' {
  if (risk >= 9) return 'critical';
  if (risk >= 7) return 'high';
  if (risk >= 4) return 'medium';
  return 'low';
}

// Legacy hooks for backward compatibility
export function useCurrentAlerts() {
  const { alerts, loadingStates, errors, refreshData } = useDashboard();
  
  return {
    data: alerts,
    loading: loadingStates.alerts,
    error: errors.alerts,
    refetch: refreshData,
  };
}

export function useSevenDayForecast() {
  // Forecast endpoint removed - return empty data
  return {
    data: null,
    loading: false,
    error: null,
    refetch: () => {},
  };
}

export function useModelPerformance() {
  // Model performance endpoint removed - return empty data
  return {
    data: null,
    loading: false,
    error: null,
    refetch: () => {},
  };
}

export function useActiveIncidents() {
  // Map alerts to incident-like structure for backward compatibility
  const { alerts, loadingStates, errors, refreshData } = useDashboard();
  
    const incidents = alerts.map(alert => ({
      id: alert.id,
      type: alert.event,
      severity: alert.severity.toLowerCase(),
      location: alert.areas.join(', '),
      description: alert.description,
      timestamp: alert.onset,
      status: 'active',
    }));
  
  return {
    data: incidents,
    loading: loadingStates.alerts,
    error: errors.alerts,
    refetch: refreshData,
  };
}

export function useSeverityByRegion() {
  const { alerts, riskPrediction, loadingStates, errors } = useDashboard();
  
  // Create region severity data based on current risk and alerts
  const regionData = alerts.length > 0 ? alerts.map(alert => ({
    region: alert.areas[0] || 'Unknown',
    severity: alert.severity.toLowerCase(),
    riskScore: riskPrediction?.overall_risk || 0,
    alertCount: 1,
  })) : [];
  
  return {
    data: regionData,
    loading: loadingStates.alerts || loadingStates.riskPrediction,
    error: errors.alerts || errors.riskPrediction,
    refetch: () => {}, // Use main dashboard refresh
  };
}