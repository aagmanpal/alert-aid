import React, { useState, useMemo } from 'react';
import styled from 'styled-components';
import { Bell, X, CheckCheck } from 'lucide-react';
import { useNotifications } from '../../contexts/NotificationContext';
import { NotificationFilter, Notification } from '../../types/notifications';
import { Card, Button, Heading, Text, Flex, Input } from '../../styles/components';

// Notification Center Container
const NotificationCenter = styled(Card)`
  position: fixed;
  top: 70px;
  right: 24px;
  width: 420px;
  max-height: 600px;
  z-index: 9998;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: ${({ theme }) => theme.shadows.lg}, 0 8px 32px rgba(0, 0, 0, 0.12);
`;

const NotificationHeader = styled(Flex)`
  padding: ${({ theme }) => theme.spacing[4]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  background: ${({ theme }) => theme.colors.background.secondary};
`;

const NotificationFilters = styled.div`
  padding: ${({ theme }) => theme.spacing[3]} ${({ theme }) => theme.spacing[4]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  background: ${({ theme }) => theme.colors.background.primary};
`;

const FilterRow = styled(Flex)`
  gap: ${({ theme }) => theme.spacing[2]};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const FilterChip = styled.button<{ active: boolean }>`
  padding: ${({ theme }) => theme.spacing[1]} ${({ theme }) => theme.spacing[3]};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  border: 1px solid ${({ theme, active }) => 
    active ? theme.colors.primary[500] : theme.colors.border.default};
  background: ${({ theme, active }) => 
    active ? theme.colors.primary[50] : theme.colors.background.primary};
  color: ${({ theme, active }) => 
    active ? theme.colors.primary[800] : theme.colors.text.secondary};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  cursor: pointer;
  transition: ${({ theme }) => theme.transitions.normal};

  &:hover {
    background: ${({ theme, active }) => 
      active ? theme.colors.primary[100] : theme.colors.surface.hover};
  }
`;

const NotificationList = styled.div`
  flex: 1;
  overflow-y: auto;
  max-height: 400px;
`;

const NotificationItem = styled.div<{ isRead: boolean }>`
  padding: ${({ theme }) => theme.spacing[4]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  background: ${({ theme, isRead }) => 
    isRead ? theme.colors.background.primary : theme.colors.background.secondary};
  transition: ${({ theme }) => theme.transitions.normal};
  cursor: pointer;

  &:hover {
    background: ${({ theme }) => theme.colors.surface.hover};
  }

  &:last-child {
    border-bottom: none;
  }
`;

const NotificationItemHeader = styled(Flex)`
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const NotificationBadge = styled.div<{ type: string }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${({ type, theme }) => {
    switch (type) {
      case 'success': return theme.colors.success[500];
      case 'error': return theme.colors.danger[500];
      case 'warning': return theme.colors.warning[500];
      case 'info': return '#3b82f6';
      default: return theme.colors.primary[500];
    }
  }};
  flex-shrink: 0;
`;

const NotificationMeta = styled(Flex)`
  margin-top: ${({ theme }) => theme.spacing[2]};
`;

const EmptyState = styled.div`
  padding: ${({ theme }) => theme.spacing[8]};
  text-align: center;
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

const NotificationActions = styled(Flex)`
  padding: ${({ theme }) => theme.spacing[3]} ${({ theme }) => theme.spacing[4]};
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
  background: ${({ theme }) => theme.colors.background.secondary};
`;

interface NotificationCenterPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const NotificationCenterPanel: React.FC<NotificationCenterPanelProps> = ({
  isOpen,
  onClose
}) => {
  const { notifications, markAsRead, markAllAsRead, clearAll, getUnreadCount } = useNotifications();
  const [filter, setFilter] = useState<NotificationFilter>({});
  const [searchTerm, setSearchTerm] = useState('');

  // Filter and search notifications
  const filteredNotifications = useMemo(() => {
    let filtered = [...notifications];

    // Apply filters
    if (filter.type && filter.type.length > 0) {
      filtered = filtered.filter(n => filter.type?.includes(n.type));
    }

    if (filter.priority && filter.priority.length > 0) {
      filtered = filtered.filter(n => filter.priority?.includes(n.priority || 'normal'));
    }

    if (filter.read !== undefined) {
      filtered = filtered.filter(n => n.read === filter.read);
    }

    // Apply search
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(n => 
        n.title.toLowerCase().includes(term) ||
        n.message?.toLowerCase().includes(term) ||
        n.source?.toLowerCase().includes(term)
      );
    }

    // Sort by timestamp (newest first)
    return filtered.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }, [notifications, filter, searchTerm]);

  const handleNotificationClick = (notification: Notification) => {
    if (!notification.read) {
      markAsRead(notification.id);
    }
  };

  const toggleFilter = (key: keyof NotificationFilter, value: any) => {
    setFilter(prev => {
      if (key === 'read') {
        return { ...prev, [key]: prev[key] === value ? undefined : value };
      }
      
      const currentArray = prev[key] as any[] || [];
      const newArray = currentArray.includes(value)
        ? currentArray.filter(item => item !== value)
        : [...currentArray, value];
      
      return { ...prev, [key]: newArray.length > 0 ? newArray : undefined };
    });
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    
    return timestamp.toLocaleDateString();
  };

  if (!isOpen) return null;

  return (
    <NotificationCenter>
      <NotificationHeader justify="between" align="center">
        <Flex align="center" gap="12px">
          <Bell size={20} />
          <Heading level={4} weight="semibold">Notifications</Heading>
          {getUnreadCount() > 0 && (
            <NotificationBadge type="info" style={{ marginLeft: '8px' }} />
          )}
        </Flex>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X size={16} />
        </Button>
      </NotificationHeader>

      <NotificationFilters>
        <FilterRow>
          <Input
            placeholder="🔍 Search notifications..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            size="sm"
          />
        </FilterRow>
        
        <FilterRow>
          <FilterChip
            active={filter.read === false}
            onClick={() => toggleFilter('read', false)}
          >
            Unread
          </FilterChip>
          <FilterChip
            active={filter.type?.includes('error') || false}
            onClick={() => toggleFilter('type', 'error')}
          >
            Errors
          </FilterChip>
          <FilterChip
            active={filter.type?.includes('warning') || false}
            onClick={() => toggleFilter('type', 'warning')}
          >
            Warnings
          </FilterChip>
          <FilterChip
            active={filter.priority?.includes('critical') || false}
            onClick={() => toggleFilter('priority', 'critical')}
          >
            Critical
          </FilterChip>
        </FilterRow>
      </NotificationFilters>

      <NotificationList>
        {filteredNotifications.length > 0 ? (
          filteredNotifications.map((notification) => (
            <NotificationItem
              key={notification.id}
              isRead={notification.read || false}
              onClick={() => handleNotificationClick(notification)}
            >
              <NotificationItemHeader justify="between" align="center">
                <Flex align="center" gap="8px">
                  <NotificationBadge type={notification.type} />
                  <Text size="sm" weight="medium">{notification.title}</Text>
                </Flex>
                <Text size="xs" color="tertiary">
                  {formatTimestamp(notification.timestamp)}
                </Text>
              </NotificationItemHeader>
              
              {notification.message && (
                <Text size="xs" color="secondary">
                  {notification.message}
                </Text>
              )}

              <NotificationMeta justify="between" align="center">
                {notification.source && (
                  <Text size="xs" color="muted">
                    {notification.source}
                  </Text>
                )}
                {notification.priority && notification.priority !== 'normal' && (
                  <Text size="xs" color="secondary">
                    {notification.priority.toUpperCase()}
                  </Text>
                )}
              </NotificationMeta>
            </NotificationItem>
          ))
        ) : (
          <EmptyState>
            <Bell size={32} style={{ marginBottom: '16px', opacity: 0.5 }} />
            <Text size="sm" color="tertiary">
              {searchTerm || Object.keys(filter).length > 0 
                ? 'No notifications match your filters'
                : 'No notifications yet'
              }
            </Text>
          </EmptyState>
        )}
      </NotificationList>

      {notifications.length > 0 && (
        <NotificationActions justify="between" align="center">
          <Flex gap="8px">
            <Button
              variant="ghost"
              size="sm"
              onClick={markAllAsRead}
              disabled={getUnreadCount() === 0}
            >
              <CheckCheck size={14} />
              Mark all read
            </Button>
          </Flex>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={clearAll}
          >
            Clear all
          </Button>
        </NotificationActions>
      )}
    </NotificationCenter>
  );
};

// Bell icon with notification indicator
const NotificationBellContainer = styled.div`
  position: relative;
  cursor: pointer;
`;

const NotificationIndicator = styled.div<{ count: number }>`
  position: absolute;
  top: -4px;
  right: -4px;
  background: ${({ theme }) => theme.colors.danger[500]};
  color: white;
  border-radius: 50%;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: bold;
  transform: ${({ count }) => count > 0 ? 'scale(1)' : 'scale(0)'};
  transition: transform 0.2s ease;
`;

interface NotificationBellProps {
  onClick: () => void;
}

export const NotificationBell: React.FC<NotificationBellProps> = ({ onClick }) => {
  const { getUnreadCount } = useNotifications();
  const unreadCount = getUnreadCount();

  return (
    <NotificationBellContainer onClick={onClick}>
      <Bell size={20} />
      <NotificationIndicator count={unreadCount}>
        {unreadCount > 99 ? '99+' : unreadCount}
      </NotificationIndicator>
    </NotificationBellContainer>
  );
};

export default NotificationCenterPanel;