import React from 'react';
import { Map, Layers, Zap, Maximize } from 'lucide-react';
import { Card, CardHeader, CardContent, Badge, IconWrapper, MetricValue } from './CardStyles';

interface IncidentMarker {
  type: string;
  lat: number;
  lng: number;
  severity: string;
}

interface MapCardProps {
  mapData: any;
}

const MapCard: React.FC<MapCardProps> = ({ mapData }) => {
  const [activeLayer, setActiveLayer] = React.useState('risk');

  const layers = [
    { id: 'risk', name: 'Risk Zones', color: '#f56565', active: activeLayer === 'risk' },
    { id: 'weather', name: 'Weather', color: '#4fd1c7', active: activeLayer === 'weather' },
    { id: 'evacuation', name: 'Evacuation', color: '#68d391', active: activeLayer === 'evacuation' },
    { id: 'resources', name: 'Resources', color: '#fbd38d', active: activeLayer === 'resources' }
  ];

  const incidentMarkers: IncidentMarker[] = mapData?.incidents || [
    { type: 'flood', lat: 34.0522, lng: -118.2437, severity: 'high' },
    { type: 'fire', lat: 34.0622, lng: -118.2537, severity: 'critical' },
    { type: 'earthquake', lat: 34.0422, lng: -118.2337, severity: 'medium' }
  ];

  return (
    <Card variant="info" size="large">
      <CardHeader>
        <h3>
          <IconWrapper variant="info">
            <Map size={20} />
          </IconWrapper>
          Interactive Risk Map
        </h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <Badge variant="info">Live Data</Badge>
          <button style={{
            background: 'transparent',
            border: '1px solid rgba(76, 209, 199, 0.4)',
            borderRadius: '6px',
            color: '#4fd1c7',
            padding: '0.25rem 0.5rem',
            fontSize: '0.7rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem'
          }}>
            <Maximize size={12} />
            Fullscreen
          </button>
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Layer Controls */}
        <div style={{
          display: 'flex',
          gap: '0.5rem',
          marginBottom: '1.5rem',
          flexWrap: 'wrap'
        }}>
          {layers.map((layer) => (
            <button
              key={layer.id}
              onClick={() => setActiveLayer(layer.id)}
              style={{
                padding: '0.5rem 0.75rem',
                background: layer.active 
                  ? `linear-gradient(135deg, ${layer.color} 0%, ${layer.color}cc 100%)`
                  : 'rgba(45, 55, 72, 0.6)',
                border: `1px solid ${layer.active ? layer.color : 'rgba(255, 255, 255, 0.1)'}`,
                borderRadius: '8px',
                color: layer.active ? '#ffffff' : '#cbd5e0',
                fontSize: '0.8rem',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem'
              }}
            >
              <Layers size={14} />
              {layer.name}
            </button>
          ))}
        </div>

        {/* Map Visualization */}
        <div style={{
          position: 'relative',
          height: '300px',
          background: 'linear-gradient(135deg, #1a202c 0%, #2d3748 100%)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          overflow: 'hidden',
          marginBottom: '1.5rem'
        }}>
          {/* Map Background Pattern */}
          <div style={{
            position: 'absolute',
            inset: 0,
            background: `
              radial-gradient(circle at 20% 30%, rgba(76, 209, 199, 0.1) 0%, transparent 50%),
              radial-gradient(circle at 80% 70%, rgba(245, 101, 101, 0.1) 0%, transparent 50%),
              radial-gradient(circle at 60% 20%, rgba(104, 211, 145, 0.1) 0%, transparent 50%),
              linear-gradient(45deg, rgba(255,255,255,0.02) 1px, transparent 1px),
              linear-gradient(-45deg, rgba(255,255,255,0.02) 1px, transparent 1px)
            `,
            backgroundSize: '100px 100px, 120px 120px, 80px 80px, 20px 20px, 20px 20px'
          }} />

          {/* Risk Zones Overlay */}
          {activeLayer === 'risk' && (
            <>
              <div style={{
                position: 'absolute',
                top: '20%',
                left: '15%',
                width: '30%',
                height: '25%',
                background: 'radial-gradient(circle, rgba(245, 101, 101, 0.3) 0%, rgba(245, 101, 101, 0.1) 70%, transparent 100%)',
                borderRadius: '50%',
                animation: 'pulse 3s ease-in-out infinite'
              }} />
              <div style={{
                position: 'absolute',
                top: '50%',
                right: '20%',
                width: '25%',
                height: '20%',
                background: 'radial-gradient(circle, rgba(251, 211, 141, 0.3) 0%, rgba(251, 211, 141, 0.1) 70%, transparent 100%)',
                borderRadius: '50%',
                animation: 'pulse 3s ease-in-out infinite 1s'
              }} />
            </>
          )}

          {/* Weather Layer */}
          {activeLayer === 'weather' && (
            <>
              <div style={{
                position: 'absolute',
                top: '10%',
                left: '10%',
                right: '10%',
                height: '30%',
                background: 'linear-gradient(90deg, rgba(76, 209, 199, 0.2) 0%, rgba(76, 209, 199, 0.1) 50%, rgba(76, 209, 199, 0.2) 100%)',
                borderRadius: '8px',
                animation: 'moveWeather 5s ease-in-out infinite'
              }} />
            </>
          )}

          {/* Evacuation Routes */}
          {activeLayer === 'evacuation' && (
            <>
              <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
                <path
                  d="M 50 50 Q 150 100 250 80 T 350 120"
                  stroke="#68d391"
                  strokeWidth="3"
                  fill="none"
                  strokeDasharray="5,5"
                  style={{ animation: 'flowRoute 3s linear infinite' }}
                />
                <path
                  d="M 80 200 Q 180 150 280 180 T 380 160"
                  stroke="#4fd1c7"
                  strokeWidth="3"
                  fill="none"
                  strokeDasharray="5,5"
                  style={{ animation: 'flowRoute 3s linear infinite 0.5s' }}
                />
              </svg>
            </>
          )}

          {/* Resource Points */}
          {activeLayer === 'resources' && (
            <>
              {[
                { x: 25, y: 30, type: 'medical' },
                { x: 65, y: 50, type: 'supply' },
                { x: 40, y: 75, type: 'shelter' },
                { x: 80, y: 25, type: 'medical' }
              ].map((point, index) => (
                <div
                  key={index}
                  style={{
                    position: 'absolute',
                    left: `${point.x}%`,
                    top: `${point.y}%`,
                    width: '12px',
                    height: '12px',
                    background: point.type === 'medical' ? '#f56565' : point.type === 'supply' ? '#fbd38d' : '#68d391',
                    borderRadius: '50%',
                    border: '2px solid #ffffff',
                    animation: `bounce 2s ease-in-out infinite ${index * 0.3}s`
                  }}
                />
              ))}
            </>
          )}

          {/* Incident Markers */}
          {incidentMarkers.map((incident, index) => (
            <div
              key={index}
              style={{
                position: 'absolute',
                left: `${20 + index * 25}%`,
                top: `${30 + index * 15}%`,
                transform: 'translate(-50%, -50%)'
              }}
            >
              <div style={{
                width: '20px',
                height: '20px',
                background: incident.severity === 'critical' ? '#f56565' : incident.severity === 'high' ? '#ed8936' : '#fbd38d',
                borderRadius: '50%',
                border: '2px solid #ffffff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                animation: 'pulse 2s ease-in-out infinite'
              }}>
                <Zap size={10} color="#ffffff" />
              </div>
            </div>
          ))}

          {/* Map Controls */}
          <div style={{
            position: 'absolute',
            top: '1rem',
            right: '1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem'
          }}>
            <button style={{
              width: '32px',
              height: '32px',
              background: 'rgba(45, 55, 72, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '6px',
              color: '#cbd5e0',
              fontSize: '1.2rem',
              fontWeight: 'bold',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              +
            </button>
            <button style={{
              width: '32px',
              height: '32px',
              background: 'rgba(45, 55, 72, 0.9)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '6px',
              color: '#cbd5e0',
              fontSize: '1.2rem',
              fontWeight: 'bold',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              −
            </button>
          </div>
        </div>

        {/* Map Statistics */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '1rem',
          marginBottom: '1.5rem'
        }}>
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#f56565' }}>
              {incidentMarkers.filter(i => i.severity === 'critical').length}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Critical</p>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#ed8936' }}>
              {incidentMarkers.filter(i => i.severity === 'high').length}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>High Risk</p>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#fbd38d' }}>
              {incidentMarkers.filter(i => i.severity === 'medium').length}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Medium</p>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#68d391' }}>
              12
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Safe Zones</p>
          </div>
        </div>

        {/* Legend */}
        <div style={{
          padding: '1rem',
          background: 'rgba(45, 55, 72, 0.6)',
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.8rem', color: '#cbd5e0' }}>Legend</h4>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '12px', height: '12px', background: '#f56565', borderRadius: '50%' }} />
              <span style={{ fontSize: '0.7rem', color: '#a0aec0' }}>Critical Incidents</span>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '12px', height: '12px', background: '#68d391', borderRadius: '50%' }} />
              <span style={{ fontSize: '0.7rem', color: '#a0aec0' }}>Safe Areas</span>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '16px', height: '2px', background: '#4fd1c7' }} />
              <span style={{ fontSize: '0.7rem', color: '#a0aec0' }}>Evacuation Routes</span>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '12px', height: '12px', background: '#fbd38d', borderRadius: '50%' }} />
              <span style={{ fontSize: '0.7rem', color: '#a0aec0' }}>Resource Centers</span>
            </div>
          </div>
        </div>

        {/* CSS Animations */}
        <style>{`
          @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.7; }
            50% { transform: scale(1.1); opacity: 1; }
          }
          
          @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
          }
          
          @keyframes moveWeather {
            0% { transform: translateX(-10px); }
            50% { transform: translateX(10px); }
            100% { transform: translateX(-10px); }
          }
          
          @keyframes flowRoute {
            0% { stroke-dashoffset: 10; }
            100% { stroke-dashoffset: 0; }
          }
        `}</style>
      </CardContent>
    </Card>
  );
};

export default MapCard;