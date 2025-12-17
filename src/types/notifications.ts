// Notification system types and interfaces

export type NotificationType = 'success' | 'warning' | 'error' | 'info';
export type NotificationPosition = 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message?: string;
  duration?: number; // milliseconds, 0 = persistent
  dismissible?: boolean;
  actions?: NotificationAction[];
  timestamp: Date;
  read?: boolean;
  priority?: 'low' | 'normal' | 'high' | 'critical';
  source?: string;
  metadata?: Record<string, any>;
}

export interface NotificationAction {
  label: string;
  action: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
}

export interface ToastConfig {
  position: NotificationPosition;
  maxVisible: number;
  defaultDuration: number;
  pauseOnHover: boolean;
  showProgress: boolean;
  stackDirection: 'up' | 'down';
  animations: {
    enter: string;
    exit: string;
    duration: number;
  };
}

export interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  addToast: (toast: { type: NotificationType; title: string; message?: string; duration?: number }) => string;
  removeNotification: (id: string) => void;
  clearAll: () => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  getUnreadCount: () => number;
  config: ToastConfig;
  updateConfig: (config: Partial<ToastConfig>) => void;
}

// Real-time alert types
export interface RealTimeAlert extends Notification {
  alertType: 'disaster' | 'system' | 'model' | 'maintenance';
  severity: 'low' | 'medium' | 'high' | 'critical';
  location?: {
    name: string;
    coordinates?: {
      lat: number;
      lng: number;
    };
  };
  affectedAreas?: string[];
  estimatedImpact?: string;
  actionRequired?: string;
  relatedIncidents?: string[];
}

// Notification center types
export interface NotificationFilter {
  type?: NotificationType[];
  priority?: ('low' | 'normal' | 'high' | 'critical')[];
  source?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  read?: boolean;
}

export interface NotificationStats {
  total: number;
  unread: number;
  byType: Record<NotificationType, number>;
  byPriority: Record<string, number>;
  todayCount: number;
  weekCount: number;
}