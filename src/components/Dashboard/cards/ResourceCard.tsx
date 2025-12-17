import React from 'react';
import { Package, Droplets, Heart, MapPin } from 'lucide-react';
import { Card, CardHeader, CardContent, MetricValue, ProgressRing, Badge, IconWrapper, Grid } from './CardStyles';

interface ResourceCardProps {
  resources: any;
}

const ResourceCard: React.FC<ResourceCardProps> = ({ resources }) => {
  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'good': return 'success';
      case 'moderate': return 'warning';
      case 'low': return 'danger';
      case 'ready': return 'success';
      default: return 'info';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return '#68d391';
      case 'moderate': return '#fbd38d';
      case 'low': return '#f56565';
      case 'ready': return '#4fd1c7';
      default: return '#a0aec0';
    }
  };

  const ResourceItem: React.FC<{
    icon: React.ReactNode;
    label: string;
    available: number;
    total: number;
    status: string;
    unit?: string;
  }> = ({ icon, label, available, total, status, unit = 'units' }) => {
    const percentage = total > 0 ? (available / total) * 100 : 0;
    
    return (
      <div style={{
        padding: '1.5rem',
        background: 'linear-gradient(135deg, rgba(26, 35, 50, 0.8) 0%, rgba(45, 55, 72, 0.6) 100%)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '16px',
        backdropFilter: 'blur(10px)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <IconWrapper variant={getStatusVariant(status)}>
            {icon}
          </IconWrapper>
          <Badge variant={getStatusVariant(status)}>
            {status}
          </Badge>
        </div>
        
        <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <ProgressRing 
              percentage={percentage} 
              size={80} 
              strokeWidth={6}
              color={getStatusColor(status)}
            >
              <svg width={80} height={80}>
                <circle
                  className="progress-ring-background"
                  cx={40}
                  cy={40}
                  r={34}
                />
                <circle
                  className="progress-ring-progress"
                  cx={40}
                  cy={40}
                  r={34}
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
              <MetricValue size="small" style={{ color: getStatusColor(status), fontSize: '1.2rem' }}>
                {available}
              </MetricValue>
              <div style={{ fontSize: '0.7rem', color: '#a0aec0' }}>/{total}</div>
            </div>
          </div>
        </div>
        
        <h4 style={{ margin: '0', fontSize: '0.9rem', color: '#e2e8f0', textAlign: 'center' }}>
          {label}
        </h4>
        <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#a0aec0', textAlign: 'center' }}>
          {unit} available
        </p>
      </div>
    );
  };

  return (
    <Card variant="info" size="medium">
      <CardHeader>
        <h3>
          <IconWrapper variant="info">
            <Package size={20} />
          </IconWrapper>
          Resource Management
        </h3>
        <Badge variant="success">All Systems Active</Badge>
      </CardHeader>
      
      <CardContent>
        <Grid columns={3} gap="1rem" style={{ marginBottom: '2rem' }}>
          <ResourceItem
            icon={<Droplets size={18} />}
            label="Water Supplies"
            available={resources?.water?.available || 85}
            total={resources?.water?.total || 100}
            status={resources?.water?.status || 'good'}
            unit="reserves"
          />
          
          <ResourceItem
            icon={<Heart size={18} />}
            label="Medical Kits"
            available={resources?.medical?.available || 67}
            total={resources?.medical?.total || 80}
            status={resources?.medical?.status || 'moderate'}
            unit="kits"
          />
          
          <ResourceItem
            icon={<MapPin size={18} />}
            label="Evacuation Points"
            available={resources?.evacuation?.points || 12}
            total={resources?.evacuation?.capacity || 5000}
            status={resources?.evacuation?.status || 'ready'}
            unit="locations"
          />
        </Grid>

        {/* Resource Allocation Chart */}
        <div style={{
          padding: '1.5rem',
          background: 'rgba(76, 209, 199, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(76, 209, 199, 0.2)',
          marginBottom: '1rem'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#4fd1c7' }}>
            📊 Resource Allocation
          </h4>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: '#cbd5e0' }}>Emergency Response</span>
            <span style={{ fontSize: '0.8rem', color: '#4fd1c7' }}>75%</span>
          </div>
          <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', marginBottom: '1rem' }}>
            <div style={{ width: '75%', height: '100%', background: '#4fd1c7', borderRadius: '3px' }} />
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: '#cbd5e0' }}>Prevention & Prep</span>
            <span style={{ fontSize: '0.8rem', color: '#fbd38d' }}>60%</span>
          </div>
          <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', marginBottom: '1rem' }}>
            <div style={{ width: '60%', height: '100%', background: '#fbd38d', borderRadius: '3px' }} />
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: '#cbd5e0' }}>Recovery Support</span>
            <span style={{ fontSize: '0.8rem', color: '#68d391' }}>90%</span>
          </div>
          <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px' }}>
            <div style={{ width: '90%', height: '100%', background: '#68d391', borderRadius: '3px' }} />
          </div>
        </div>

        {/* Quick Actions */}
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button style={{
            flex: 1,
            padding: '0.75rem',
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
            Request Supplies
          </button>
          
          <button style={{
            flex: 1,
            padding: '0.75rem',
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
            View Inventory
          </button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ResourceCard;