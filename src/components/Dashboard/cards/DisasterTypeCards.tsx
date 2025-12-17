import React from 'react';
import { Waves, Zap, Mountain, Flame } from 'lucide-react';
import { Card, CardHeader, CardContent, MetricValue, ProgressRing, Badge, IconWrapper, Grid } from './CardStyles';

interface DisasterTypeCardsProps {
  prediction: any;
}

const DisasterTypeCard: React.FC<{
  type: string;
  icon: React.ReactNode;
  risk: number;
  trend?: 'up' | 'down' | 'stable';
}> = ({ type, icon, risk, trend = 'stable' }) => {
  const getRiskColor = (score: number) => {
    if (score >= 8) return '#f56565';
    if (score >= 6) return '#fbd38d';
    if (score >= 4) return '#fbd38d';
    return '#68d391';
  };

  const getRiskVariant = (score: number) => {
    if (score >= 8) return 'danger';
    if (score >= 6) return 'warning';
    if (score >= 4) return 'warning';
    return 'success';
  };

  const getTrendSymbol = () => {
    switch (trend) {
      case 'up': return '↗';
      case 'down': return '↘';
      default: return '→';
    }
  };

  return (
    <div style={{ 
      background: 'linear-gradient(135deg, rgba(26, 35, 50, 0.8) 0%, rgba(45, 55, 72, 0.6) 100%)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      borderRadius: '16px',
      padding: '1.5rem',
      backdropFilter: 'blur(10px)',
      transition: 'all 0.3s ease',
      cursor: 'pointer'
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.transform = 'translateY(-2px)';
      e.currentTarget.style.borderColor = 'rgba(76, 209, 199, 0.3)';
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.transform = 'translateY(0)';
      e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <IconWrapper variant={getRiskVariant(risk)}>
          {icon}
        </IconWrapper>
        <Badge variant={getRiskVariant(risk)}>
          {getTrendSymbol()}
        </Badge>
      </div>
      
      <div style={{ textAlign: 'center' }}>
        <div style={{ position: 'relative', display: 'inline-block', marginBottom: '1rem' }}>
          <ProgressRing 
            percentage={(risk / 10) * 100} 
            size={80} 
            strokeWidth={6}
            color={getRiskColor(risk)}
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
            <MetricValue size="small" style={{ color: getRiskColor(risk), fontSize: '1.5rem' }}>
              {risk.toFixed(1)}
            </MetricValue>
          </div>
        </div>
        
        <h4 style={{ margin: '0', fontSize: '1rem', color: '#e2e8f0', fontWeight: '600' }}>
          {type}
        </h4>
        <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#a0aec0' }}>
          Risk Level
        </p>
      </div>
    </div>
  );
};

const DisasterTypeCards: React.FC<DisasterTypeCardsProps> = ({ prediction }) => {
  const disasterTypes = [
    {
      type: 'Flood Risk',
      icon: <Waves size={20} />,
      risk: prediction?.flood_risk || 0,
      trend: 'stable' as const
    },
    {
      type: 'Storm Risk',
      icon: <Zap size={20} />,
      risk: prediction?.storm_risk || 0,
      trend: 'up' as const
    },
    {
      type: 'Earthquake Risk',
      icon: <Mountain size={20} />,
      risk: prediction?.earthquake_risk || 0,
      trend: 'stable' as const
    },
    {
      type: 'Wildfire Risk',
      icon: <Flame size={20} />,
      risk: prediction?.fire_risk || 0,
      trend: 'down' as const
    }
  ];

  return (
    <Card size="medium" style={{ gridColumn: 'span 2' }}>
      <CardHeader>
        <h3>Active Disaster Types</h3>
        <Badge variant="info">Live Monitoring</Badge>
      </CardHeader>
      
      <CardContent>
        <Grid columns={2} gap="1.5rem">
          {disasterTypes.map((disaster, index) => (
            <DisasterTypeCard
              key={index}
              type={disaster.type}
              icon={disaster.icon}
              risk={disaster.risk}
              trend={disaster.trend}
            />
          ))}
        </Grid>
        
        <div style={{ 
          marginTop: '1.5rem', 
          padding: '1rem', 
          background: 'rgba(76, 209, 199, 0.1)', 
          borderRadius: '12px',
          border: '1px solid rgba(76, 209, 199, 0.2)'
        }}>
          <p style={{ margin: '0', fontSize: '0.9rem', color: '#4fd1c7', fontWeight: '500' }}>
            💡 Risk Assessment
          </p>
          <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: '#a0aec0' }}>
            Based on real-time weather data, historical patterns, and ML predictions. 
            Updates every 5 minutes.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default DisasterTypeCards;