import React from 'react';
import styled from 'styled-components';
import { AlertTriangle, CheckCircle, XCircle, Info, Zap } from 'lucide-react';
import { Card, Button, Heading, Text } from '../../styles/components';
import { useNotifications } from '../../contexts/NotificationContext';

const DemoContainer = styled(Card)`
  padding: ${({ theme }) => theme.spacing[6]};
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const ButtonGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${({ theme }) => theme.spacing[3]};
  margin-top: ${({ theme }) => theme.spacing[4]};
`;

const NotificationDemo: React.FC = () => {
  const { addNotification, addToast } = useNotifications();

  const demoNotifications = [
    {
      type: 'error' as const,
      title: 'Critical Earthquake Alert',
      message: 'Magnitude 7.2 earthquake detected near San Francisco. Immediate evacuation recommended.',
      priority: 'critical' as const,
      source: 'USGS Earthquake Monitor',
      actions: [
        { label: 'View Map', action: () => console.log('Opening earthquake map') },
        { label: 'Emergency Protocol', action: () => console.log('Activating emergency protocol') }
      ]
    },
    {
      type: 'warning' as const,
      title: 'Flood Warning',
      message: 'Heavy rainfall expected. Flash flood warning for riverside areas.',
      priority: 'high' as const,
      source: 'Weather Service',
      actions: [
        { label: 'Safety Tips', action: () => console.log('Showing flood safety tips') },
        { label: 'Evacuation Routes', action: () => console.log('Showing evacuation routes') }
      ]
    },
    {
      type: 'success' as const,
      title: 'System Status',
      message: 'All monitoring systems are operational and functioning normally.',
      priority: 'normal' as const,
      source: 'System Monitor'
    },
    {
      type: 'info' as const,
      title: 'Weekly Report Ready',
      message: 'Your weekly disaster preparedness report is ready for review.',
      priority: 'normal' as const,
      source: 'Report Generator'
    }
  ];

  const demoToasts = [
    {
      type: 'success' as const,
      title: 'Action Completed',
      message: 'Emergency response team has been notified successfully.'
    },
    {
      type: 'error' as const,
      title: 'Connection Failed',
      message: 'Unable to connect to emergency services. Please check your connection.'
    },
    {
      type: 'warning' as const,
      title: 'Battery Low',
      message: 'Emergency radio battery is running low. Please recharge.'
    },
    {
      type: 'info' as const,
      title: 'System Update',
      message: 'New safety features have been added to your dashboard.'
    }
  ];

  return (
    <DemoContainer>
      <Heading level={4} weight="semibold">Notification System Demo</Heading>
      <Text size="sm" color="secondary" style={{ marginTop: '8px' }}>
        Test the Material Design notification system with realistic emergency alert scenarios.
      </Text>
      
      <ButtonGrid>
        <div>
          <Text size="sm" weight="medium" style={{ marginBottom: '12px' }}>
            📢 Notification Center Alerts
          </Text>
          {demoNotifications.map((notification, index) => (
            <Button
              key={index}
              variant="outline"
              size="sm"
              onClick={() => addNotification(notification)}
              style={{ 
                width: '100%',
                marginBottom: '8px',
                justifyContent: 'flex-start',
                gap: '8px'
              }}
            >
              {notification.type === 'error' && <XCircle size={16} />}
              {notification.type === 'warning' && <AlertTriangle size={16} />}
              {notification.type === 'success' && <CheckCircle size={16} />}
              {notification.type === 'info' && <Info size={16} />}
              {notification.title}
            </Button>
          ))}
        </div>

        <div>
          <Text size="sm" weight="medium" style={{ marginBottom: '12px' }}>
            🍞 Toast Notifications
          </Text>
          {demoToasts.map((toast, index) => (
            <Button
              key={index}
              variant="outline"
              size="sm"
              onClick={() => addToast(toast)}
              style={{ 
                width: '100%',
                marginBottom: '8px',
                justifyContent: 'flex-start',
                gap: '8px'
              }}
            >
              {toast.type === 'error' && <XCircle size={16} />}
              {toast.type === 'warning' && <AlertTriangle size={16} />}
              {toast.type === 'success' && <CheckCircle size={16} />}
              {toast.type === 'info' && <Info size={16} />}
              {toast.title}
            </Button>
          ))}
        </div>

        <div>
          <Text size="sm" weight="medium" style={{ marginBottom: '12px' }}>
            ⚡ Quick Actions
          </Text>
          
          <Button
            variant="primary"
            size="sm"
            onClick={() => {
              // Simulate multiple emergency alerts
              addNotification({
                type: 'error',
                title: 'EMERGENCY: Wildfire Detected',
                message: 'Fast-moving wildfire 5 miles northeast. Immediate evacuation required for zones A-C.',
                priority: 'critical',
                source: 'Fire Department',
                actions: [
                  { label: 'Evacuation Map', action: () => addToast({ type: 'info', title: 'Opening evacuation map...', message: '' }) },
                  { label: 'Call 911', action: () => addToast({ type: 'success', title: 'Emergency services contacted', message: '' }) }
                ]
              });
              
              setTimeout(() => {
                addToast({
                  type: 'warning',
                  title: 'Air Quality Alert',
                  message: 'Smoke detected. Stay indoors and close windows.'
                });
              }, 1000);
              
              setTimeout(() => {
                addToast({
                  type: 'info',
                  title: 'Traffic Update',
                  message: 'Highway 101 closed. Use alternate routes.'
                });
              }, 2000);
            }}
            style={{ width: '100%', marginBottom: '8px' }}
          >
            <Zap size={16} />
            Simulate Emergency
          </Button>
          
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              // Multiple success notifications
              addToast({ type: 'success', title: 'Systems Online', message: 'All monitoring systems operational' });
              addToast({ type: 'success', title: 'Teams Deployed', message: 'Emergency response teams in position' });
              addToast({ type: 'success', title: 'Alerts Sent', message: 'Emergency alerts sent to 50,000 residents' });
            }}
            style={{ width: '100%', marginBottom: '8px' }}
          >
            <CheckCircle size={16} />
            All Systems Go
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              addNotification({
                type: 'info',
                title: 'System Maintenance',
                message: 'Scheduled maintenance will occur tonight from 2-4 AM PST.',
                priority: 'normal',
                source: 'IT Operations'
              });
            }}
            style={{ width: '100%' }}
          >
            <Info size={16} />
            Maintenance Notice
          </Button>
        </div>
      </ButtonGrid>
    </DemoContainer>
  );
};

export default NotificationDemo;