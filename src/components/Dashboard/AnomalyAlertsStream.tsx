import React, { useState, useEffect } from 'react';
import styled, { keyframes, css } from 'styled-components';
import { Card, Text, Flex } from '../../styles/components';
import { AlertTriangle, Zap, TrendingUp, Activity, Clock } from 'lucide-react';

const slideIn = keyframes`
  from {
    opacity: 0;
    transform: translateX(-10px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
`;

const pulse = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
`;

const AlertsContainer = styled(Card)`
  min-height: 320px;
  display: flex;
  flex-direction: column;
  background: ${({ theme }) => theme.colors.surface.default};
  border: 2px solid ${({ theme }) => theme.colors.primary[100]}; /* Coral border */
  border-radius: ${({ theme }) => theme.borderRadius.xl}; /* More rounded */
  box-shadow: ${({ theme }) => theme.shadows.warmGlow}; /* Warm glow */
`;

const AlertsHeader = styled(Flex)`
  margin-bottom: ${({ theme }) => theme.spacing[4]};
  padding-bottom: ${({ theme }) => theme.spacing[3]};
  border-bottom: 2px solid ${({ theme }) => theme.colors.primary[200]}; /* Coral separator */
`;

const AlertsList = styled.div`
  flex: 1;
  overflow-y: auto;
  max-height: 240px;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
`;

const AlertItem = styled.div<{ severity: 'low' | 'medium' | 'high' | 'critical'; isNew?: boolean }>`
  display: flex;
  align-items: flex-start;
  gap: ${({ theme }) => theme.spacing.sm};
  padding: ${({ theme }) => theme.spacing.sm};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background: ${({ severity }) => {
    switch (severity) {
      case 'critical': return '#fef2f2';
      case 'high': return '#fff7ed';
      case 'medium': return '#f0f9ff';
      case 'low': return '#f0fdf4';
      default: return '#ffffff';
    }
  }};
  border: 1px solid ${({ severity }) => {
    switch (severity) {
      case 'critical': return '#fecaca';
      case 'high': return '#fed7aa';
      case 'medium': return '#bae6fd';
      case 'low': return '#bbf7d0';
      default: return '#e2e8f0';
    }
  }};
  border-left: 3px solid ${({ severity }) => {
    switch (severity) {
      case 'critical': return '#dc2626';
      case 'high': return '#ea580c';
      case 'medium': return '#0ea5e9';
      case 'low': return '#16a34a';
      default: return '#64748b';
    }
  }};
  
  ${({ isNew }) => isNew && css`
    animation: ${slideIn} 0.3s ease-out;
  `}
  
  transition: all 0.2s ease;
  
  &:hover {
    background: ${({ severity }) => {
      switch (severity) {
        case 'critical': return '#fee2e2';
        case 'high': return '#ffedd5';
        case 'medium': return '#e0f2fe';
        case 'low': return '#dcfce7';
        default: return '#f8fafc';
      }
    }};
    transform: translateX(2px);
    box-shadow: 0 2px 4px 0 rgb(0 0 0 / 0.1);
  }
`;

const AlertIcon = styled.div<{ severity: 'low' | 'medium' | 'high' | 'critical' }>`
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 2px;
  background: ${({ severity }) => {
    switch (severity) {
      case 'critical': return '#dc2626';
      case 'high': return '#ea580c';
      case 'medium': return '#0ea5e9';
      case 'low': return '#16a34a';
      default: return '#64748b';
    }
  }};
  
  ${({ severity }) => severity === 'critical' && css`
    animation: ${pulse} 1.5s infinite;
  `}
`;

const AlertContent = styled.div`
  flex: 1;
  min-width: 0;
`;

const AlertTime = styled(Text)`
  font-size: 10px;
  opacity: 0.6;
  margin-top: 2px;
`;

const LiveIndicator = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #16a34a;
  ${css`animation: ${pulse} 2s infinite;`}
`;

// Mock anomaly alerts data
const mockAnomalyAlerts = [
  {
    id: 1,
    type: 'seismic_anomaly',
    severity: 'critical' as const,
    message: 'Unusual seismic pattern detected in San Andreas Fault zone',
    location: 'California, USA',
    confidence: 94,
    timestamp: new Date(Date.now() - 2 * 60 * 1000), // 2 mins ago
    model: 'SeismicNet-v2.1'
  },
  {
    id: 2,
    type: 'weather_anomaly',
    severity: 'high' as const,
    message: 'Rapid atmospheric pressure drop indicating severe storm formation',
    location: 'Gulf of Mexico',
    confidence: 87,
    timestamp: new Date(Date.now() - 8 * 60 * 1000), // 8 mins ago
    model: 'WeatherML-v3.0'
  },
  {
    id: 3,
    type: 'flood_risk',
    severity: 'medium' as const,
    message: 'Rising river levels with potential flood risk in 6-8 hours',
    location: 'Mississippi River Basin',
    confidence: 78,
    timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 mins ago
    model: 'FloodPredict-v1.8'
  },
  {
    id: 4,
    type: 'wildfire_risk',
    severity: 'high' as const,
    message: 'Extreme fire weather conditions with low humidity and high winds',
    location: 'Northern California',
    confidence: 91,
    timestamp: new Date(Date.now() - 22 * 60 * 1000), // 22 mins ago
    model: 'FireRisk-v2.3'
  },
  {
    id: 5,
    type: 'volcanic_activity',
    severity: 'low' as const,
    message: 'Minor volcanic tremors detected, monitoring continued',
    location: 'Mount Rainier, WA',
    confidence: 65,
    timestamp: new Date(Date.now() - 35 * 60 * 1000), // 35 mins ago
    model: 'VolcanoWatch-v1.5'
  }
];

interface AnomalyAlert {
  id: number;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  location: string;
  confidence: number;
  timestamp: Date;
  model: string;
}

interface AnomalyAlertsStreamProps {
  alerts?: AnomalyAlert[];
}

const AnomalyAlertsStream: React.FC<AnomalyAlertsStreamProps> = ({ 
  alerts = mockAnomalyAlerts 
}) => {
  const [displayAlerts, setDisplayAlerts] = useState(alerts);
  const [newAlertIds, setNewAlertIds] = useState<Set<number>>(new Set());

  useEffect(() => {
    // Simulate real-time alerts
    const interval = setInterval(() => {
      const shouldAddAlert = Math.random() > 0.7; // 30% chance of new alert
      
      if (shouldAddAlert) {
        const newAlert: AnomalyAlert = {
          id: Date.now(),
          type: ['seismic_anomaly', 'weather_anomaly', 'flood_risk'][Math.floor(Math.random() * 3)],
          severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any,
          message: 'Real-time anomaly detected by ML models',
          location: 'Various Locations',
          confidence: Math.floor(Math.random() * 30) + 70,
          timestamp: new Date(),
          model: 'RealTimeML-v1.0'
        };
        
        setDisplayAlerts(prev => [newAlert, ...prev.slice(0, 9)]); // Keep max 10 alerts
        setNewAlertIds(prev => {
          const updated = new Set(prev);
          updated.add(newAlert.id);
          return updated;
        });
        
        // Remove "new" status after 3 seconds
        setTimeout(() => {
          setNewAlertIds(prev => {
            const updated = new Set(prev);
            updated.delete(newAlert.id);
            return updated;
          });
        }, 3000);
      }
    }, 10000); // Check every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'seismic_anomaly': return <Activity size={12} />;
      case 'weather_anomaly': return <Zap size={12} />;
      case 'flood_risk': return <TrendingUp size={12} />;
      case 'volcanic_activity': return <AlertTriangle size={12} />;
      case 'wildfire_risk': return <AlertTriangle size={12} />;
      default: return <AlertTriangle size={12} />;
    }
  };

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  const criticalCount = displayAlerts.filter(alert => alert.severity === 'critical').length;
  const highCount = displayAlerts.filter(alert => alert.severity === 'high').length;

  return (
    <AlertsContainer>
      <AlertsHeader justify="between" align="center">
        <div>
          <Flex align="center" gap="8px">
            <Text size="lg" weight="semibold">Anomaly Alerts</Text>
            <LiveIndicator />
          </Flex>
          <Text size="sm" color="secondary" style={{ marginTop: '2px' }}>
            Real-time ML anomaly detection stream
          </Text>
        </div>
        <Flex align="center" gap="12px">
          {criticalCount > 0 && (
            <Flex align="center" gap="4px">
              <div style={{ 
                width: '8px', 
                height: '8px', 
                borderRadius: '50%', 
                background: '#f44336' 
              }} />
              <Text size="xs" weight="medium">{criticalCount} Critical</Text>
            </Flex>
          )}
          {highCount > 0 && (
            <Flex align="center" gap="4px">
              <div style={{ 
                width: '8px', 
                height: '8px', 
                borderRadius: '50%', 
                background: '#ff9800' 
              }} />
              <Text size="xs" weight="medium">{highCount} High</Text>
            </Flex>
          )}
        </Flex>
      </AlertsHeader>

      <AlertsList>
        {displayAlerts.map((alert) => (
          <AlertItem 
            key={alert.id} 
            severity={alert.severity}
            isNew={newAlertIds.has(alert.id)}
          >
            <AlertIcon severity={alert.severity}>
              {getTypeIcon(alert.type)}
            </AlertIcon>
            
            <AlertContent>
              <Text size="sm" weight="medium">{alert.message}</Text>
              <Flex align="center" gap="8px" style={{ marginTop: '4px' }}>
                <Text size="xs" color="secondary">{alert.location}</Text>
                <Text size="xs" color="secondary">•</Text>
                <Text size="xs" weight="medium" style={{ color: '#4caf50' }}>
                  {alert.confidence}% confidence
                </Text>
              </Flex>
              <Flex align="center" gap="8px" style={{ marginTop: '2px' }}>
                <Clock size={10} style={{ opacity: 0.6 }} />
                <AlertTime color="secondary">{formatTimeAgo(alert.timestamp)}</AlertTime>
                <Text size="xs" color="secondary">•</Text>
                <Text size="xs" color="secondary">{alert.model}</Text>
              </Flex>
            </AlertContent>
          </AlertItem>
        ))}
      </AlertsList>
    </AlertsContainer>
  );
};

export default AnomalyAlertsStream;