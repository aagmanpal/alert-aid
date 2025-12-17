import React from 'react';
import styled from 'styled-components';
import { createPortal } from 'react-dom';
import { useNotifications } from '../../contexts/NotificationContext';
import Toast from './Toast';

// Position mapping
const getPositionStyles = (position: string) => {
  const base = {
    position: 'fixed' as const,
    zIndex: 9999,
    pointerEvents: 'none' as const,
  };

  switch (position) {
    case 'top-right':
      return { ...base, top: '24px', right: '24px' };
    case 'top-left':
      return { ...base, top: '24px', left: '24px' };
    case 'bottom-right':
      return { ...base, bottom: '24px', right: '24px' };
    case 'bottom-left':
      return { ...base, bottom: '24px', left: '24px' };
    case 'top-center':
      return { ...base, top: '24px', left: '50%', transform: 'translateX(-50%)' };
    case 'bottom-center':
      return { ...base, bottom: '24px', left: '50%', transform: 'translateX(-50%)' };
    default:
      return { ...base, top: '24px', right: '24px' };
  }
};

const ToastPortalContainer = styled.div<{ position: string; stackDirection: string }>`
  ${({ position }) => getPositionStyles(position)}
  display: flex;
  flex-direction: ${({ stackDirection }) => stackDirection === 'up' ? 'column-reverse' : 'column'};
  gap: 8px;
  max-height: 100vh;
  overflow: hidden;
  
  & > * {
    pointer-events: auto;
  }
`;

interface ToastContainerProps {
  portalRoot?: Element;
}

export const ToastContainer: React.FC<ToastContainerProps> = ({ 
  portalRoot = document.body 
}) => {
  const { notifications, removeNotification, config } = useNotifications();

  // Filter notifications for toast display (show only recent ones)
  const toastNotifications = notifications
    .filter((n: any) => {
      // Only show notifications that should be displayed as toasts
      const isRecent = Date.now() - n.timestamp.getTime() < 300000; // 5 minutes
      return isRecent || n.duration === 0; // Keep persistent notifications
    })
    .slice(-config.maxVisible); // Limit visible toasts

  if (toastNotifications.length === 0) {
    return null;
  }

  const containerContent = (
    <ToastPortalContainer
      position={config.position}
      stackDirection={config.stackDirection}
    >
      {toastNotifications.map((notification: any) => (
        <Toast
          key={notification.id}
          notification={notification}
          onDismiss={removeNotification}
          position={config.position}
          showProgress={config.showProgress}
          pauseOnHover={config.pauseOnHover}
        />
      ))}
    </ToastPortalContainer>
  );

  return createPortal(containerContent, portalRoot);
};

export default ToastContainer;