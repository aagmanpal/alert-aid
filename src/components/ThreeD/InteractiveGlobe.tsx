/**
 * INTERACTIVE 3D GLOBE - Production Component
 * Real-time disaster risk visualization with live data integration
 */

import React, { useEffect, useRef, useState } from 'react';
import Globe from 'react-globe.gl';
import styled from 'styled-components';
import { useLocation } from '../../contexts/LocationContext';

interface RiskPoint {
  lat: number;
  lng: number;
  size: number;
  color: string;
  label: string;
  severity: 'low' | 'moderate' | 'high' | 'critical';
}

interface InteractiveGlobeProps {
  alerts?: any[];
  riskScore?: number;
  width?: number;
  height?: number;
}

const GlobeContainer = styled.div`
  position: relative;
  width: 100%;
  height: 100%;
  border-radius: 12px;
  overflow: hidden;
  background: ${({ theme }) => theme.colors.background.secondary};
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
`;

const GlobeControls = styled.div`
  position: absolute;
  top: 16px;
  right: 16px;
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const ControlButton = styled.button`
  background: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  color: ${({ theme }) => theme.colors.text.primary};
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    background: ${({ theme }) => theme.colors.surface.hover};
    border-color: ${({ theme }) => theme.colors.border.accent};
  }
`;

const LiveIndicator = styled.div`
  position: absolute;
  top: 16px;
  left: 16px;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 8px;
  background: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 0.875rem;
  color: ${({ theme }) => theme.colors.text.primary};
`;

const PulseDot = styled.span`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #22C55E;
  animation: pulse 2s ease-in-out infinite;

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
`;

const InteractiveGlobe: React.FC<InteractiveGlobeProps> = ({ 
  alerts = [], 
  riskScore = 0,
  width = 600, 
  height = 600 
}) => {
  const globeRef = useRef<any>(null);
  const { currentLocation } = useLocation();
  const [riskPoints, setRiskPoints] = useState<RiskPoint[]>([]);
  const [autoRotate, setAutoRotate] = useState(true);

  useEffect(() => {
    // Convert alerts to risk points
    const points: RiskPoint[] = alerts.map((alert: any) => {
      const severity = alert.severity?.toLowerCase() || 'moderate';
      
      return {
        lat: alert.location?.latitude || alert.lat || 0,
        lng: alert.location?.longitude || alert.lon || 0,
        size: getSizeForSeverity(severity),
        color: getColorForSeverity(severity),
        label: alert.title || alert.type || 'Alert',
        severity: severity as 'low' | 'moderate' | 'high' | 'critical'
      };
    });

    // Add user location marker
    if (currentLocation) {
      points.push({
        lat: currentLocation.latitude,
        lng: currentLocation.longitude,
        size: 1.2,
        color: '#3B82F6',
        label: 'Your Location',
        severity: 'moderate'
      });
    }

    setRiskPoints(points);
  }, [alerts, currentLocation]);

  useEffect(() => {
    // Auto-rotate to user location
    if (globeRef.current && currentLocation) {
      globeRef.current.pointOfView(
        {
          lat: currentLocation.latitude,
          lng: currentLocation.longitude,
          altitude: 2.5
        },
        1500
      );
    }
  }, [currentLocation]);

  const getSizeForSeverity = (severity: string): number => {
    switch (severity) {
      case 'critical': return 2.0;
      case 'high': return 1.5;
      case 'moderate': return 1.0;
      default: return 0.7;
    }
  };

  const getColorForSeverity = (severity: string): string => {
    switch (severity) {
      case 'critical': return '#DC2626';
      case 'high': return '#EF4444';
      case 'moderate': return '#F59E0B';
      default: return '#22C55E';
    }
  };

  const resetView = () => {
    if (globeRef.current && currentLocation) {
      globeRef.current.pointOfView(
        {
          lat: currentLocation.latitude,
          lng: currentLocation.longitude,
          altitude: 2.5
        },
        1000
      );
    }
  };

  return (
    <GlobeContainer>
      <LiveIndicator>
        <PulseDot />
        <span>Live Data</span>
      </LiveIndicator>

      <GlobeControls>
        <ControlButton onClick={() => setAutoRotate(!autoRotate)}>
          {autoRotate ? '⏸ Pause' : '▶ Rotate'}
        </ControlButton>
        <ControlButton onClick={resetView}>
          🎯 Reset View
        </ControlButton>
      </GlobeControls>

      <Globe
        ref={globeRef}
        globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
        backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
        
        // Risk points
        pointsData={riskPoints}
        pointLat={(d: any) => d.lat}
        pointLng={(d: any) => d.lng}
        pointAltitude={0.01}
        pointRadius={(d: any) => d.size}
        pointColor={(d: any) => d.color}
        pointLabel={(d: any) => d.label}
        
        // Atmosphere
        atmosphereColor="#3B82F6"
        atmosphereAltitude={0.15}
        
        // Controls
        width={width}
        height={height}
        enablePointerInteraction={true}
      />
    </GlobeContainer>
  );
};

export default InteractiveGlobe;
