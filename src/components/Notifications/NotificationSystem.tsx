import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';
import styled from 'styled-components';
// Import will be added when we integrate real-time data
// import { useRealTimeData, AlertData } from '../../hooks/useRealTimeData';

// Types for notification system
export interface DisasterAlert {
  id: string;
  type: 'earthquake' | 'flood' | 'wildfire' | 'storm' | 'general';
  severity: 'low' | 'medium' | 'high' | 'critical';
  riskLevel: number; // 0-10 scale
  title: string;
  message: string;
  location: string;
  timestamp: number;
  expiresAt?: number;
  actions?: AlertAction[];
  isDismissed: boolean;
  isRead: boolean;
  source: string;
}

export interface AlertAction {
  label: string;
  action: 'dismiss' | 'mark-safe' | 'view-details' | 'share' | 'call-emergency';
  variant: 'primary' | 'secondary' | 'danger';
}

interface NotificationState {
  alerts: DisasterAlert[];
  activePopup: DisasterAlert | null;
  lastPopupTime: number;
  unreadCount: number;
  settings: NotificationSettings;
}

interface NotificationSettings {
  enablePopups: boolean;
  minPopupRiskLevel: number; // 9-10 for critical only
  popupCooldown: number; // 30 minutes default
  enableSounds: boolean;
  quietHours: { start: string; end: string } | null;
}

// Notification Context
const NotificationContext = createContext<{
  state: NotificationState;
  addAlert: (alert: Omit<DisasterAlert, 'id' | 'timestamp' | 'isDismissed' | 'isRead'>) => void;
  dismissAlert: (id: string) => void;
  markAsRead: (id: string) => void;
  markAsSafe: (id: string) => void;
  clearAll: () => void;
  updateSettings: (settings: Partial<NotificationSettings>) => void;
} | null>(null);

// Styled components for notification system
const NotificationPopup = styled.div<{ isVisible: boolean }>`
  position: fixed;
  top: ${({ theme }) => theme.spacing.lg};
  right: ${({ theme }) => theme.spacing.lg};
  width: 400px;
  max-width: calc(100vw - ${({ theme }) => theme.spacing.xl});
  background: ${({ theme }) => theme.colors.surface.elevated};
  border: 2px solid ${({ theme }) => theme.colors.primary[600]};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.xl};
  padding: ${({ theme }) => theme.spacing.lg};
  z-index: 1100;
  transform: translateX(${({ isVisible }) => isVisible ? '0' : '120%'});
  transition: transform 0.3s ease-in-out;
  backdrop-filter: blur(10px);
  
  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    top: ${({ theme }) => theme.spacing.md};
    right: ${({ theme }) => theme.spacing.md};
    left: ${({ theme }) => theme.spacing.md};
    width: auto;
  }
`;

const PopupHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const PopupTitle = styled.h3<{ severity: string }>`
  color: ${({ theme, severity }) => 
    severity === 'critical' ? theme.colors.danger[600] :
    severity === 'high' ? theme.colors.warning[600] :
    theme.colors.text.primary
  };
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
`;

const PopupClose = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 1.2rem;
  cursor: pointer;
  padding: ${({ theme }) => theme.spacing.xs};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  
  &:hover {
    background: ${({ theme }) => theme.colors.surface.hover};
    color: ${({ theme }) => theme.colors.text.primary};
  }
`;

const PopupContent = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.lg};
`;

const PopupMessage = styled.p`
  color: ${({ theme }) => theme.colors.text.secondary};
  margin: ${({ theme }) => theme.spacing.sm} 0;
  line-height: 1.5;
`;

const PopupMeta = styled.div`
  font-size: 0.875rem;
  color: ${({ theme }) => theme.colors.text.caption};
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: ${({ theme }) => theme.spacing.sm};
`;

const PopupActions = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.sm};
  flex-wrap: wrap;
`;

const ActionButton = styled.button<{ variant: 'primary' | 'secondary' | 'danger' }>`
  padding: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.md};
  border: none;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  
  background: ${({ theme, variant }) => 
    variant === 'primary' ? theme.colors.primary[600] :
    variant === 'danger' ? theme.colors.danger[600] :
    theme.colors.surface.hover
  };
  
  color: ${({ theme, variant }) => 
    variant === 'secondary' ? theme.colors.text.primary : theme.colors.text.inverse
  };
  
  &:hover {
    transform: translateY(-1px);
    background: ${({ theme, variant }) => 
      variant === 'primary' ? theme.colors.primary[700] :
      variant === 'danger' ? theme.colors.danger[700] :
      theme.colors.surface.border
    };
  }
`;

// Severity icons
const getSeverityIcon = (severity: string) => {
  switch (severity) {
    case 'critical': return '🚨';
    case 'high': return '⚠️';
    case 'medium': return '📍';
    case 'low': return 'ℹ️';
    default: return '📢';
  }
};

// Notification Provider Component
export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<NotificationState>({
    alerts: [],
    activePopup: null,
    lastPopupTime: 0,
    unreadCount: 0,
    settings: {
      enablePopups: true,
      minPopupRiskLevel: 9, // Only show popups for critical alerts (9-10)
      popupCooldown: 30 * 60 * 1000, // 30 minutes
      enableSounds: true,
      quietHours: null
    }
  });

  // Generate unique alert ID
  const generateId = () => `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  // Check if we're in quiet hours
  const isQuietHours = useCallback(() => {
    if (!state.settings.quietHours) return false;
    
    const now = new Date();
    const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    const { start, end } = state.settings.quietHours;
    
    if (start <= end) {
      return currentTime >= start && currentTime <= end;
    } else {
      // Quiet hours span midnight
      return currentTime >= start || currentTime <= end;
    }
  }, [state.settings.quietHours]);

  // Add new alert
  const addAlert = useCallback((alertData: Omit<DisasterAlert, 'id' | 'timestamp' | 'isDismissed' | 'isRead'>) => {
    const newAlert: DisasterAlert = {
      ...alertData,
      id: generateId(),
      timestamp: Date.now(),
      isDismissed: false,
      isRead: false
    };

    setState(prev => {
      const updatedAlerts = [newAlert, ...prev.alerts];
      const shouldShowPopup = 
        prev.settings.enablePopups &&
        newAlert.riskLevel >= prev.settings.minPopupRiskLevel &&
        (Date.now() - prev.lastPopupTime) >= prev.settings.popupCooldown &&
        !isQuietHours();

      return {
        ...prev,
        alerts: updatedAlerts,
        activePopup: shouldShowPopup ? newAlert : prev.activePopup,
        lastPopupTime: shouldShowPopup ? Date.now() : prev.lastPopupTime,
        unreadCount: prev.unreadCount + 1
      };
    });

    // Play notification sound if enabled
    if (state.settings.enableSounds && !isQuietHours()) {
      playNotificationSound(alertData.severity);
    }
  }, [isQuietHours, state.settings.enableSounds]);

  // Play notification sound
  const playNotificationSound = (severity: string) => {
    try {
      // Create different tones for different severities
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      // Different frequencies for different severities
      const frequency = severity === 'critical' ? 880 : severity === 'high' ? 660 : 440;
      oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
      
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
      
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.5);
    } catch (error) {
      console.log('Audio notification failed:', error);
    }
  };

  // Dismiss alert
  const dismissAlert = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      alerts: prev.alerts.map(alert => 
        alert.id === id ? { ...alert, isDismissed: true } : alert
      ),
      activePopup: prev.activePopup?.id === id ? null : prev.activePopup
    }));
  }, []);

  // Mark alert as read
  const markAsRead = useCallback((id: string) => {
    setState(prev => {
      const alert = prev.alerts.find(a => a.id === id);
      if (!alert || alert.isRead) return prev;

      return {
        ...prev,
        alerts: prev.alerts.map(alert => 
          alert.id === id ? { ...alert, isRead: true } : alert
        ),
        unreadCount: Math.max(0, prev.unreadCount - 1)
      };
    });
  }, []);

  // Mark as safe (user confirms they're safe)
  const markAsSafe = useCallback((id: string) => {
    // Store safety confirmation
    const safetyReport = {
      alertId: id,
      timestamp: Date.now(),
      status: 'safe'
    };
    
    const existingReports = JSON.parse(localStorage.getItem('safetyReports') || '[]');
    localStorage.setItem('safetyReports', JSON.stringify([safetyReport, ...existingReports]));
    
    dismissAlert(id);
  }, [dismissAlert]);

  // Clear all alerts
  const clearAll = useCallback(() => {
    setState(prev => ({
      ...prev,
      alerts: [],
      activePopup: null,
      unreadCount: 0
    }));
  }, []);

  // Update settings
  const updateSettings = useCallback((newSettings: Partial<NotificationSettings>) => {
    setState(prev => ({
      ...prev,
      settings: { ...prev.settings, ...newSettings }
    }));
  }, []);

  // Handle alert actions
  const handleAction = useCallback((alert: DisasterAlert, action: AlertAction) => {
    switch (action.action) {
      case 'dismiss':
        dismissAlert(alert.id);
        break;
      case 'mark-safe':
        markAsSafe(alert.id);
        break;
      case 'view-details':
        // Navigate to detailed view
        markAsRead(alert.id);
        break;
      case 'share':
        if (navigator.share) {
          navigator.share({
            title: alert.title,
            text: alert.message,
            url: window.location.href
          });
        }
        break;
      case 'call-emergency':
        window.open('tel:911');
        break;
    }
  }, [dismissAlert, markAsSafe, markAsRead]);

  // Auto-dismiss popup after 10 seconds
  useEffect(() => {
    if (state.activePopup) {
      const timer = setTimeout(() => {
        setState(prev => ({ ...prev, activePopup: null }));
      }, 10000);
      
      return () => clearTimeout(timer);
    }
  }, [state.activePopup]);

  // Clean up old alerts (remove alerts older than 24 hours)
  useEffect(() => {
    const cleanup = () => {
      const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
      setState(prev => ({
        ...prev,
        alerts: prev.alerts.filter(alert => 
          alert.timestamp > oneDayAgo && !alert.isDismissed
        )
      }));
    };

    const timer = setInterval(cleanup, 60 * 60 * 1000); // Clean up every hour
    return () => clearInterval(timer);
  }, []);

  return (
    <NotificationContext.Provider value={{
      state,
      addAlert,
      dismissAlert,
      markAsRead,
      markAsSafe,
      clearAll,
      updateSettings
    }}>
      {children}
      
      {/* Notification Popup */}
      {state.activePopup && (
        <NotificationPopup isVisible={!!state.activePopup}>
          <PopupHeader>
            <PopupTitle severity={state.activePopup.severity}>
              {getSeverityIcon(state.activePopup.severity)}
              {state.activePopup.title}
            </PopupTitle>
            <PopupClose onClick={() => dismissAlert(state.activePopup!.id)}>
              ✕
            </PopupClose>
          </PopupHeader>
          
          <PopupContent>
            <PopupMessage>{state.activePopup.message}</PopupMessage>
            <PopupMeta>
              <span>📍 {state.activePopup.location}</span>
              <span>Risk: {state.activePopup.riskLevel}/10</span>
            </PopupMeta>
          </PopupContent>
          
          {state.activePopup.actions && (
            <PopupActions>
              {state.activePopup.actions.map((action, index) => (
                <ActionButton
                  key={index}
                  variant={action.variant}
                  onClick={() => handleAction(state.activePopup!, action)}
                >
                  {action.label}
                </ActionButton>
              ))}
            </PopupActions>
          )}
        </NotificationPopup>
      )}
    </NotificationContext.Provider>
  );
};

// Hook to use notification system
export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
};