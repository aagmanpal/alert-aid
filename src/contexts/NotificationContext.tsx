import React, { createContext, useContext, useReducer, useCallback, ReactNode } from 'react';
import { Notification, NotificationContextType, ToastConfig, NotificationType } from '../types/notifications';

// Default configuration
const defaultConfig: ToastConfig = {
  position: 'top-right',
  maxVisible: 5,
  defaultDuration: 5000,
  pauseOnHover: true,
  showProgress: true,
  stackDirection: 'down',
  animations: {
    enter: 'slideInRight',
    exit: 'slideOutRight',
    duration: 300
  }
};

// Notification actions
type NotificationAction = 
  | { type: 'ADD_NOTIFICATION'; payload: Notification }
  | { type: 'REMOVE_NOTIFICATION'; payload: string }
  | { type: 'CLEAR_ALL' }
  | { type: 'MARK_AS_READ'; payload: string }
  | { type: 'MARK_ALL_AS_READ' }
  | { type: 'UPDATE_CONFIG'; payload: Partial<ToastConfig> };

// Notification state
interface NotificationState {
  notifications: Notification[];
  config: ToastConfig;
}

// Initial state
const initialState: NotificationState = {
  notifications: [],
  config: defaultConfig
};

// Reducer
function notificationReducer(state: NotificationState, action: NotificationAction): NotificationState {
  switch (action.type) {
    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [...state.notifications, action.payload]
      };
    
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload)
      };
    
    case 'CLEAR_ALL':
      return {
        ...state,
        notifications: []
      };
    
    case 'MARK_AS_READ':
      return {
        ...state,
        notifications: state.notifications.map(n => 
          n.id === action.payload ? { ...n, read: true } : n
        )
      };
    
    case 'MARK_ALL_AS_READ':
      return {
        ...state,
        notifications: state.notifications.map(n => ({ ...n, read: true }))
      };
    
    case 'UPDATE_CONFIG':
      return {
        ...state,
        config: { ...state.config, ...action.payload }
      };
    
    default:
      return state;
  }
}

// Create context
const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// Provider component
interface NotificationProviderProps {
  children: ReactNode;
  config?: Partial<ToastConfig>;
}

export function NotificationProvider({ children, config }: NotificationProviderProps) {
  const [state, dispatch] = useReducer(notificationReducer, {
    ...initialState,
    config: { ...defaultConfig, ...config }
  });

  // Generate unique ID
  const generateId = useCallback(() => {
    return `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Add notification
  const addNotification = useCallback((notificationData: Omit<Notification, 'id' | 'timestamp'>) => {
    const id = generateId();
    const notification: Notification = {
      id,
      timestamp: new Date(),
      duration: state.config.defaultDuration,
      dismissible: true,
      read: false,
      priority: 'normal',
      ...notificationData
    };

    dispatch({ type: 'ADD_NOTIFICATION', payload: notification });

    // Auto-remove if duration is set
    if (notification.duration && notification.duration > 0) {
      setTimeout(() => {
        dispatch({ type: 'REMOVE_NOTIFICATION', payload: id });
      }, notification.duration);
    }

    return id;
  }, [generateId, state.config.defaultDuration]);

  // Add toast (simplified notification for temporary messages)
  const addToast = useCallback((toast: { type: NotificationType; title: string; message?: string; duration?: number }) => {
    return addNotification({
      type: toast.type,
      title: toast.title,
      message: toast.message,
      duration: toast.duration || 4000,
      dismissible: true,
      priority: 'normal'
    });
  }, [addNotification]);

  // Remove notification
  const removeNotification = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_NOTIFICATION', payload: id });
  }, []);

  // Clear all notifications
  const clearAll = useCallback(() => {
    dispatch({ type: 'CLEAR_ALL' });
  }, []);

  // Mark as read
  const markAsRead = useCallback((id: string) => {
    dispatch({ type: 'MARK_AS_READ', payload: id });
  }, []);

  // Mark all as read
  const markAllAsRead = useCallback(() => {
    dispatch({ type: 'MARK_ALL_AS_READ' });
  }, []);

  // Get unread count
  const getUnreadCount = useCallback(() => {
    return state.notifications.filter(n => !n.read).length;
  }, [state.notifications]);

  // Update config
  const updateConfig = useCallback((newConfig: Partial<ToastConfig>) => {
    dispatch({ type: 'UPDATE_CONFIG', payload: newConfig });
  }, []);

  const contextValue: NotificationContextType = {
    notifications: state.notifications,
    addNotification,
    addToast,
    removeNotification,
    clearAll,
    markAsRead,
    markAllAsRead,
    getUnreadCount,
    config: state.config,
    updateConfig
  };

  return (
    <NotificationContext.Provider value={contextValue}>
      {children}
    </NotificationContext.Provider>
  );
}

// Hook to use notifications
export function useNotifications() {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
}

// Convenience hooks for different notification types
export function useToast() {
  const { addNotification } = useNotifications();

  return {
    success: (title: string, message?: string, options?: Partial<Notification>) =>
      addNotification({ type: 'success', title, message, ...options }),
    
    error: (title: string, message?: string, options?: Partial<Notification>) =>
      addNotification({ type: 'error', title, message, duration: 0, ...options }),
    
    warning: (title: string, message?: string, options?: Partial<Notification>) =>
      addNotification({ type: 'warning', title, message, ...options }),
    
    info: (title: string, message?: string, options?: Partial<Notification>) =>
      addNotification({ type: 'info', title, message, ...options })
  };
}