import React from 'react';
import { Bell, AlertTriangle, Info, CheckCircle } from 'lucide-react';
import { Card, CardHeader, CardContent, Badge, IconWrapper, MetricValue } from './CardStyles';

interface AlertItem {
  title: string;
  message: string;
  priority: string;
  time: string;
  type: string;
}

interface AlertsCardProps {
  alerts: any;
}

const AlertsCard: React.FC<AlertsCardProps> = ({ alerts }) => {
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return '#f56565';
      case 'high': return '#ed8936';
      case 'medium': return '#fbd38d';
      case 'low': return '#4fd1c7';
      default: return '#a0aec0';
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'critical': return <AlertTriangle size={16} />;
      case 'high': return <Bell size={16} />;
      case 'medium': return <Info size={16} />;
      case 'low': return <CheckCircle size={16} />;
      default: return <Info size={16} />;
    }
  };

  const AlertItem: React.FC<{
    title: string;
    message: string;
    priority: string;
    time: string;
    type: string;
  }> = ({ title, message, priority, time, type }) => (
    <div style={{
      padding: '1rem',
      background: `linear-gradient(135deg, ${getPriorityColor(priority)}20 0%, ${getPriorityColor(priority)}10 100%)`,
      border: `1px solid ${getPriorityColor(priority)}40`,
      borderRadius: '12px',
      marginBottom: '0.75rem',
      transition: 'transform 0.2s ease',
      cursor: 'pointer'
    }}
    onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
    onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <IconWrapper variant={priority === 'critical' ? 'danger' : priority === 'high' ? 'warning' : 'info'}>
            {getPriorityIcon(priority)}
          </IconWrapper>
          <h4 style={{ margin: 0, fontSize: '0.9rem', color: '#e2e8f0' }}>{title}</h4>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Badge variant={priority === 'critical' ? 'danger' : priority === 'high' ? 'warning' : 'info'}>
            {priority}
          </Badge>
          <span style={{ fontSize: '0.7rem', color: '#a0aec0' }}>{time}</span>
        </div>
      </div>
      
      <p style={{ margin: '0', fontSize: '0.8rem', color: '#cbd5e0', lineHeight: '1.4' }}>
        {message}
      </p>
      
      <div style={{ marginTop: '0.5rem' }}>
        <span style={{
          fontSize: '0.7rem',
          color: getPriorityColor(priority),
          fontWeight: '600',
          textTransform: 'uppercase'
        }}>
          {type}
        </span>
      </div>
    </div>
  );

  const alertsData: AlertItem[] = alerts?.active || [
    {
      title: "Flash Flood Warning",
      message: "Heavy rainfall detected in watershed areas. River levels rising rapidly. Immediate evacuation recommended for low-lying areas.",
      priority: "critical",
      time: "2 min ago",
      type: "Weather Alert"
    },
    {
      title: "Earthquake Precursor Detected",
      message: "Seismic activity indicators show increased probability. Magnitude 4.5+ predicted within 48 hours.",
      priority: "high",
      time: "15 min ago",
      type: "Seismic Alert"
    },
    {
      title: "Wildfire Risk Elevated",
      message: "Low humidity and high winds creating dangerous fire conditions. Exercise extreme caution.",
      priority: "medium",
      time: "1 hour ago",
      type: "Fire Weather"
    },
    {
      title: "System Health Check",
      message: "All monitoring stations are operational. Data feeds are stable and current.",
      priority: "low",
      time: "3 hours ago",
      type: "System Status"
    }
  ];

  return (
    <Card variant="warning" size="large">
      <CardHeader>
        <h3>
          <IconWrapper variant="warning">
            <Bell size={20} />
          </IconWrapper>
          Active Alerts
        </h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <Badge variant="danger">{alertsData.filter(a => a.priority === 'critical').length} Critical</Badge>
          <Badge variant="warning">{alertsData.filter(a => a.priority === 'high').length} High</Badge>
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Alert Statistics */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '1rem',
          marginBottom: '2rem',
          padding: '1.5rem',
          background: 'rgba(237, 137, 54, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(237, 137, 54, 0.2)'
        }}>
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#f56565' }}>
              {alertsData.filter(a => a.priority === 'critical').length}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Critical</p>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#ed8936' }}>
              {alertsData.filter(a => a.priority === 'high').length}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>High</p>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#fbd38d' }}>
              {alertsData.filter(a => a.priority === 'medium').length}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Medium</p>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#4fd1c7' }}>
              {alertsData.filter(a => a.priority === 'low').length}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#a0aec0' }}>Low</p>
          </div>
        </div>

        {/* Active Alerts List */}
        <div style={{ 
          maxHeight: '400px', 
          overflowY: 'auto',
          paddingRight: '0.5rem',
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(255,255,255,0.3) transparent'
        }}>
          {alertsData.map((alert, index) => (
            <AlertItem
              key={index}
              title={alert.title}
              message={alert.message}
              priority={alert.priority}
              time={alert.time}
              type={alert.type}
            />
          ))}
        </div>

        {/* Response Actions */}
        <div style={{
          marginTop: '2rem',
          padding: '1.5rem',
          background: 'rgba(76, 209, 199, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(76, 209, 199, 0.2)'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#4fd1c7' }}>
            🚨 Emergency Response Center
          </h4>
          
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <button style={{
              padding: '0.75rem 1rem',
              background: 'linear-gradient(135deg, #f56565 0%, #e53e3e 100%)',
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
              Emergency Broadcast
            </button>
            
            <button style={{
              padding: '0.75rem 1rem',
              background: 'linear-gradient(135deg, #ed8936 0%, #dd6b20 100%)',
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
              Dispatch Teams
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
              View Details
            </button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default AlertsCard;