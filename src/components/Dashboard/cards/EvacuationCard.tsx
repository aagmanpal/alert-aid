import React from 'react';
import { MapPin, Navigation, Shield, Users } from 'lucide-react';
import { Card, CardHeader, CardContent, Badge, IconWrapper, MetricValue, ProgressRing } from './CardStyles';

interface EvacuationCardProps {
  evacuation: any;
}

const EvacuationCard: React.FC<EvacuationCardProps> = ({ evacuation }) => {
  const RouteItem: React.FC<{
    name: string;
    status: string;
    capacity: number;
    current: number;
    distance: string;
    eta: string;
  }> = ({ name, status, capacity, current, distance, eta }) => {
    const occupancyPercent = capacity > 0 ? (current / capacity) * 100 : 0;
    const statusColor = status === 'open' ? '#68d391' : status === 'busy' ? '#fbd38d' : '#f56565';
    
    return (
      <div style={{
        padding: '1.25rem',
        background: 'linear-gradient(135deg, rgba(26, 35, 50, 0.8) 0%, rgba(45, 55, 72, 0.6) 100%)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '12px',
        backdropFilter: 'blur(10px)',
        marginBottom: '1rem',
        transition: 'transform 0.2s ease',
        cursor: 'pointer'
      }}
      onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
      onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <IconWrapper variant={status === 'open' ? 'success' : status === 'busy' ? 'warning' : 'danger'}>
              <Navigation size={16} />
            </IconWrapper>
            <h4 style={{ margin: 0, fontSize: '0.95rem', color: '#e2e8f0' }}>{name}</h4>
          </div>
          <Badge variant={status === 'open' ? 'success' : status === 'busy' ? 'warning' : 'danger'}>
            {status}
          </Badge>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '1rem', alignItems: 'center' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', color: '#cbd5e0' }}>Capacity:</span>
              <span style={{ fontSize: '0.8rem', color: statusColor, fontWeight: '600' }}>
                {current}/{capacity} people
              </span>
            </div>
            
            <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px' }}>
              <div style={{ 
                width: `${Math.min(occupancyPercent, 100)}%`, 
                height: '100%', 
                background: statusColor, 
                borderRadius: '3px',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: '#a0aec0' }}>Distance</div>
            <div style={{ fontSize: '0.9rem', color: '#4fd1c7', fontWeight: '600' }}>{distance}</div>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: '#a0aec0' }}>ETA</div>
            <div style={{ fontSize: '0.9rem', color: '#4fd1c7', fontWeight: '600' }}>{eta}</div>
          </div>
        </div>
      </div>
    );
  };

  const ShelterItem: React.FC<{
    name: string;
    type: string;
    capacity: number;
    available: number;
    resources: string[];
  }> = ({ name, type, capacity, available, resources }) => {
    const availabilityPercent = capacity > 0 ? (available / capacity) * 100 : 0;
    
    return (
      <div style={{
        padding: '1.25rem',
        background: 'rgba(76, 209, 199, 0.1)',
        border: '1px solid rgba(76, 209, 199, 0.2)',
        borderRadius: '12px',
        marginBottom: '1rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <IconWrapper variant="info">
              <Shield size={16} />
            </IconWrapper>
            <div>
              <h4 style={{ margin: 0, fontSize: '0.9rem', color: '#e2e8f0' }}>{name}</h4>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>{type}</p>
            </div>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#4fd1c7' }}>
              {available}
            </MetricValue>
            <div style={{ fontSize: '0.7rem', color: '#a0aec0' }}>available</div>
          </div>
        </div>
        
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: '#cbd5e0' }}>Capacity</span>
            <span style={{ fontSize: '0.8rem', color: '#4fd1c7' }}>
              {Math.round(availabilityPercent)}% available
            </span>
          </div>
          <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px' }}>
            <div style={{ 
              width: `${availabilityPercent}%`, 
              height: '100%', 
              background: '#4fd1c7', 
              borderRadius: '3px' 
            }} />
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          {resources.map((resource, index) => (
            <span
              key={index}
              style={{
                padding: '0.25rem 0.5rem',
                background: 'rgba(76, 209, 199, 0.2)',
                color: '#4fd1c7',
                fontSize: '0.7rem',
                borderRadius: '6px',
                fontWeight: '500'
              }}
            >
              {resource}
            </span>
          ))}
        </div>
      </div>
    );
  };

  const evacuationData = evacuation || {
    routes: [
      { name: "Highway 101 North", status: "open", capacity: 500, current: 125, distance: "8.2 km", eta: "12 min" },
      { name: "Coastal Route West", status: "busy", capacity: 300, current: 280, distance: "6.5 km", eta: "18 min" },
      { name: "Mountain Pass East", status: "closed", capacity: 200, current: 0, distance: "12.1 km", eta: "--" }
    ],
    shelters: [
      { name: "Community Center", type: "Primary Shelter", capacity: 800, available: 600, resources: ["Medical", "Food", "Communications"] },
      { name: "High School Gymnasium", type: "Secondary Shelter", capacity: 400, available: 380, resources: ["Food", "Bedding"] },
      { name: "Sports Complex", type: "Emergency Shelter", capacity: 600, available: 450, resources: ["Medical", "Bedding", "Pet Care"] }
    ]
  };

  return (
    <Card variant="success" size="large">
      <CardHeader>
        <h3>
          <IconWrapper variant="success">
            <MapPin size={20} />
          </IconWrapper>
          Evacuation Management
        </h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <Badge variant="success">
            {evacuationData.routes.filter((r: any) => r.status === 'open').length} Routes Open
          </Badge>
          <Badge variant="info">
            {evacuationData.shelters.reduce((acc: number, s: any) => acc + s.available, 0)} Shelter Spaces
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Evacuation Overview */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '1rem',
          marginBottom: '2rem',
          padding: '1.5rem',
          background: 'rgba(104, 211, 145, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(104, 211, 145, 0.2)'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ position: 'relative', display: 'inline-block', marginBottom: '0.5rem' }}>
              <ProgressRing 
                percentage={75} 
                size={60} 
                strokeWidth={4}
                color="#68d391"
              >
                <svg width={60} height={60}>
                  <circle
                    className="progress-ring-background"
                    cx={30}
                    cy={30}
                    r={26}
                  />
                  <circle
                    className="progress-ring-progress"
                    cx={30}
                    cy={30}
                    r={26}
                  />
                </svg>
              </ProgressRing>
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center'
              }}>
                <IconWrapper variant="success" style={{ background: 'transparent', border: 'none' }}>
                  <Navigation size={16} />
                </IconWrapper>
              </div>
            </div>
            <MetricValue size="small" style={{ color: '#68d391' }}>75%</MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Routes Clear</p>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <div style={{ position: 'relative', display: 'inline-block', marginBottom: '0.5rem' }}>
              <ProgressRing 
                percentage={82} 
                size={60} 
                strokeWidth={4}
                color="#4fd1c7"
              >
                <svg width={60} height={60}>
                  <circle
                    className="progress-ring-background"
                    cx={30}
                    cy={30}
                    r={26}
                  />
                  <circle
                    className="progress-ring-progress"
                    cx={30}
                    cy={30}
                    r={26}
                  />
                </svg>
              </ProgressRing>
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center'
              }}>
                <IconWrapper variant="info" style={{ background: 'transparent', border: 'none' }}>
                  <Shield size={16} />
                </IconWrapper>
              </div>
            </div>
            <MetricValue size="small" style={{ color: '#4fd1c7' }}>1,430</MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Shelter Capacity</p>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <div style={{ position: 'relative', display: 'inline-block', marginBottom: '0.5rem' }}>
              <ProgressRing 
                percentage={100} 
                size={60} 
                strokeWidth={4}
                color="#68d391"
              >
                <svg width={60} height={60}>
                  <circle
                    className="progress-ring-background"
                    cx={30}
                    cy={30}
                    r={26}
                  />
                  <circle
                    className="progress-ring-progress"
                    cx={30}
                    cy={30}
                    r={26}
                  />
                </svg>
              </ProgressRing>
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center'
              }}>
                <IconWrapper variant="success" style={{ background: 'transparent', border: 'none' }}>
                  <Users size={16} />
                </IconWrapper>
              </div>
            </div>
            <MetricValue size="small" style={{ color: '#68d391' }}>100%</MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Systems Ready</p>
          </div>
        </div>

        {/* Evacuation Routes */}
        <div style={{ marginBottom: '2rem' }}>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '1rem', color: '#e2e8f0' }}>
            🛣️ Evacuation Routes
          </h4>
          
          {evacuationData.routes.map((route: any, index: number) => (
            <RouteItem
              key={index}
              name={route.name}
              status={route.status}
              capacity={route.capacity}
              current={route.current}
              distance={route.distance}
              eta={route.eta}
            />
          ))}
        </div>

        {/* Emergency Shelters */}
        <div>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '1rem', color: '#e2e8f0' }}>
            🏠 Emergency Shelters
          </h4>
          
          {evacuationData.shelters.map((shelter: any, index: number) => (
            <ShelterItem
              key={index}
              name={shelter.name}
              type={shelter.type}
              capacity={shelter.capacity}
              available={shelter.available}
              resources={shelter.resources}
            />
          ))}
        </div>

        {/* Emergency Actions */}
        <div style={{
          marginTop: '2rem',
          padding: '1.5rem',
          background: 'rgba(237, 137, 54, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(237, 137, 54, 0.2)'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#ed8936' }}>
            🚨 Emergency Coordination
          </h4>
          
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <button style={{
              padding: '0.75rem 1rem',
              background: 'linear-gradient(135deg, #68d391 0%, #48bb78 100%)',
              border: 'none',
              borderRadius: '8px',
              color: '#ffffff',
              fontSize: '0.85rem',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'transform 0.2s ease'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
            >
              Open New Route
            </button>
            
            <button style={{
              padding: '0.75rem 1rem',
              background: 'linear-gradient(135deg, #4fd1c7 0%, #38b2ac 100%)',
              border: 'none',
              borderRadius: '8px',
              color: '#ffffff',
              fontSize: '0.85rem',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'transform 0.2s ease'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
            >
              Activate Shelter
            </button>
            
            <button style={{
              padding: '0.75rem 1rem',
              background: 'linear-gradient(135deg, #4a5568 0%, #2d3748 100%)',
              border: 'none',
              borderRadius: '8px',
              color: '#ffffff',
              fontSize: '0.85rem',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'transform 0.2s ease'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
            >
              View Map
            </button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default EvacuationCard;