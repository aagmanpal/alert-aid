import React, { useState } from 'react';
import styled from 'styled-components';
import { Card, Button } from '../../styles/components';

// Types for communication system
interface EmergencyBroadcast {
  id: string;
  type: 'EAS' | 'WEA' | 'local' | 'weather' | 'amber';
  title: string;
  message: string;
  severity: 'info' | 'warning' | 'severe' | 'extreme';
  issuedBy: string;
  issuedAt: number;
  expiresAt?: number;
  affectedAreas: string[];
  actions: string[];
  isRead: boolean;
}

interface CommunityPost {
  id: string;
  author: string;
  content: string;
  timestamp: number;
  location?: string;
  verified: boolean;
  category: 'report' | 'help' | 'info' | 'resource' | 'safety';
  priority: 'low' | 'medium' | 'high';
  likes: number;
  reports: number;
  hasImage?: boolean;
}

interface EmergencyMessage {
  id: string;
  recipient: 'emergency-services' | 'family' | 'community';
  subject: string;
  content: string;
  status: 'draft' | 'sent' | 'delivered' | 'read';
  sentAt?: number;
  priority: 'normal' | 'high' | 'urgent';
  location?: { lat: number; lon: number };
}

// Styled components
const CommunicationContainer = styled(Card)`
  background: ${({ theme }) => theme.colors.surface.elevated};
  padding: ${({ theme }) => theme.spacing.lg};
`;

const TabBar = styled.div`
  display: flex;
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  border-bottom: 2px solid ${({ theme }) => theme.colors.surface.border};
`;

const Tab = styled.button<{ isActive: boolean }>`
  padding: ${({ theme }) => theme.spacing.md} ${({ theme }) => theme.spacing.lg};
  border: none;
  background: none;
  color: ${({ theme, isActive }) => 
    isActive ? theme.colors.primary[600] : theme.colors.text.secondary
  };
  font-weight: ${({ isActive }) => isActive ? '600' : '400'};
  border-bottom: 2px solid ${({ theme, isActive }) => 
    isActive ? theme.colors.primary[600] : 'transparent'
  };
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    color: ${({ theme }) => theme.colors.primary[600]};
  }
`;

const BroadcastList = styled.div`
  max-height: 400px;
  overflow-y: auto;
`;

const BroadcastItem = styled.div<{ severity: string; isRead: boolean }>`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 2px solid ${({ theme, severity }) => 
    severity === 'extreme' ? theme.colors.danger[400] :
    severity === 'severe' ? theme.colors.warning[400] :
    theme.colors.surface.border
  };
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.md};
  opacity: ${({ isRead }) => isRead ? 0.8 : 1};
  
  ${({ severity }) => severity === 'extreme' && `
    animation: urgentPulse 2s infinite;
    
    @keyframes urgentPulse {
      0%, 100% { border-color: rgba(215, 38, 56, 0.8); }
      50% { border-color: rgba(215, 38, 56, 1); }
    }
  `}
`;

const BroadcastHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const BroadcastInfo = styled.div`
  flex: 1;
  
  .type {
    color: ${({ theme }) => theme.colors.text.caption};
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 4px;
  }
  
  .title {
    color: ${({ theme }) => theme.colors.text.primary};
    font-weight: 600;
    margin-bottom: 4px;
  }
  
  .issuer {
    color: ${({ theme }) => theme.colors.text.secondary};
    font-size: 0.85rem;
  }
`;

const SeverityBadge = styled.span<{ severity: string }>`
  padding: 4px 12px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  
  background: ${({ theme, severity }) => 
    severity === 'extreme' ? theme.colors.danger[600] :
    severity === 'severe' ? theme.colors.warning[600] :
    severity === 'warning' ? theme.colors.warning[500] :
    theme.colors.surface.border
  };
  
  color: ${({ theme }) => theme.colors.text.inverse};
`;

const BroadcastContent = styled.div`
  margin: ${({ theme }) => theme.spacing.sm} 0;
  
  .message {
    color: ${({ theme }) => theme.colors.text.secondary};
    line-height: 1.5;
    margin-bottom: ${({ theme }) => theme.spacing.sm};
  }
  
  .areas {
    font-size: 0.85rem;
    color: ${({ theme }) => theme.colors.text.caption};
  }
`;

const CommunityFeed = styled.div`
  max-height: 400px;
  overflow-y: auto;
`;

const PostItem = styled.div<{ priority: string; verified: boolean }>`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 1px solid ${({ theme, verified, priority }) => 
    verified ? theme.colors.success[400] :
    priority === 'high' ? theme.colors.warning[400] :
    theme.colors.surface.border
  };
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const PostHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const PostAuthor = styled.div`
  .name {
    color: ${({ theme }) => theme.colors.text.primary};
    font-weight: 600;
    font-size: 0.9rem;
  }
  
  .time {
    color: ${({ theme }) => theme.colors.text.caption};
    font-size: 0.8rem;
  }
`;

const PostBadges = styled.div`
  display: flex;
  gap: 4px;
`;

const VerifiedBadge = styled.span`
  background: ${({ theme }) => theme.colors.success[600]};
  color: ${({ theme }) => theme.colors.text.inverse};
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.borderRadius.xs};
  font-size: 0.7rem;
  font-weight: 600;
`;

const CategoryBadge = styled.span<{ category: string }>`
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.borderRadius.xs};
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  
  background: ${({ theme, category }) => 
    category === 'report' ? theme.colors.warning[100] :
    category === 'help' ? theme.colors.danger[100] :
    category === 'resource' ? theme.colors.success[100] :
    theme.colors.surface.hover
  };
  
  color: ${({ theme, category }) => 
    category === 'report' ? theme.colors.warning[700] :
    category === 'help' ? theme.colors.danger[700] :
    category === 'resource' ? theme.colors.success[700] :
    theme.colors.text.primary
  };
`;

const PostContent = styled.div`
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: 1.5;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const PostActions = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing.md};
  font-size: 0.85rem;
  color: ${({ theme }) => theme.colors.text.caption};
  
  button {
    background: none;
    border: none;
    color: ${({ theme }) => theme.colors.text.caption};
    cursor: pointer;
    padding: 4px 8px;
    border-radius: ${({ theme }) => theme.borderRadius.sm};
    
    &:hover {
      background: ${({ theme }) => theme.colors.surface.hover};
      color: ${({ theme }) => theme.colors.text.secondary};
    }
  }
`;

const MessageComposer = styled.div`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 1px solid ${({ theme }) => theme.colors.surface.border};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const MessageForm = styled.div`
  display: grid;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const FormInput = styled.input`
  padding: ${({ theme }) => theme.spacing.sm};
  border: 1px solid ${({ theme }) => theme.colors.surface.border};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  background: ${({ theme }) => theme.colors.surface.hover};
  color: ${({ theme }) => theme.colors.text.primary};
  
  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary[600]};
  }
`;

const FormTextArea = styled.textarea`
  padding: ${({ theme }) => theme.spacing.sm};
  border: 1px solid ${({ theme }) => theme.colors.surface.border};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  background: ${({ theme }) => theme.colors.surface.hover};
  color: ${({ theme }) => theme.colors.text.primary};
  min-height: 80px;
  resize: vertical;
  
  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary[600]};
  }
`;

const FormSelect = styled.select`
  padding: ${({ theme }) => theme.spacing.sm};
  border: 1px solid ${({ theme }) => theme.colors.surface.border};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  background: ${({ theme }) => theme.colors.surface.hover};
  color: ${({ theme }) => theme.colors.text.primary};
  
  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.primary[600]};
  }
`;

// Main component
const CommunicationHub: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'broadcasts' | 'community' | 'messages'>('broadcasts');
  
  // Mock data - in production, fetch from APIs
  const [broadcasts, setBroadcasts] = useState<EmergencyBroadcast[]>([
    {
      id: '1',
      type: 'EAS',
      title: 'Severe Weather Warning',
      message: 'Strong winds and heavy rain expected in the area. Secure outdoor items and avoid travel if possible.',
      severity: 'severe',
      issuedBy: 'National Weather Service',
      issuedAt: Date.now() - 1800000, // 30 min ago
      expiresAt: Date.now() + 7200000, // 2 hours from now
      affectedAreas: ['Downtown', 'Riverside', 'Hill District'],
      actions: ['Secure outdoor furniture', 'Avoid unnecessary travel', 'Monitor weather updates'],
      isRead: false
    },
    {
      id: '2',
      type: 'WEA',
      title: 'Flash Flood Warning',
      message: 'Rapid water rise possible in low-lying areas. Move to higher ground immediately if flooding occurs.',
      severity: 'extreme',
      issuedBy: 'Emergency Management',
      issuedAt: Date.now() - 900000, // 15 min ago
      affectedAreas: ['Creek Valley', 'Industrial Zone'],
      actions: ['Move to higher ground', 'Avoid flooded roads', 'Call 911 if trapped'],
      isRead: false
    },
    {
      id: '3',
      type: 'local',
      title: 'Road Closure Update',
      message: 'Highway 101 northbound closed due to accident. Use alternate routes.',
      severity: 'warning',
      issuedBy: 'Traffic Management',
      issuedAt: Date.now() - 2700000, // 45 min ago
      affectedAreas: ['Highway 101'],
      actions: ['Use alternate routes', 'Allow extra travel time'],
      isRead: true
    }
  ]);

  const [communityPosts, setCommunityPosts] = useState<CommunityPost[]>([
    {
      id: '1',
      author: 'Sarah M. (City Official)',
      content: 'Evacuation center at Lincoln High is now open. Free meals and shelter available. Pet-friendly.',
      timestamp: Date.now() - 600000, // 10 min ago
      verified: true,
      category: 'resource',
      priority: 'high',
      likes: 24,
      reports: 0
    },
    {
      id: '2',
      author: 'Mike R.',
      content: 'Power lines down on Oak Street between 5th and 7th. Area blocked off by utility crews. Avoid the area.',
      timestamp: Date.now() - 1200000, // 20 min ago
      location: 'Oak Street',
      verified: false,
      category: 'report',
      priority: 'medium',
      likes: 12,
      reports: 0
    },
    {
      id: '3',
      author: 'Emergency Volunteer',
      content: 'Looking for volunteers to help distribute supplies at the community center. Contact me if you can help.',
      timestamp: Date.now() - 1800000, // 30 min ago
      verified: true,
      category: 'help',
      priority: 'medium',
      likes: 8,
      reports: 0
    }
  ]);

  const [newMessage, setNewMessage] = useState<Partial<EmergencyMessage>>({
    recipient: 'family',
    priority: 'normal',
    subject: '',
    content: ''
  });

  const [sentMessages, setSentMessages] = useState<EmergencyMessage[]>([]);

  // Mark broadcast as read
  const markBroadcastAsRead = (id: string) => {
    setBroadcasts(prev => prev.map(broadcast =>
      broadcast.id === id ? { ...broadcast, isRead: true } : broadcast
    ));
  };

  // Send emergency message
  const sendMessage = () => {
    if (!newMessage.subject || !newMessage.content) return;

    const message: EmergencyMessage = {
      id: Date.now().toString(),
      recipient: newMessage.recipient!,
      subject: newMessage.subject,
      content: newMessage.content,
      priority: newMessage.priority!,
      status: 'sent',
      sentAt: Date.now(),
      location: { lat: 37.7749, lon: -122.4194 } // Current location
    };

    setSentMessages(prev => [message, ...prev]);
    setNewMessage({
      recipient: 'family',
      priority: 'normal',
      subject: '',
      content: ''
    });

    // Simulate message delivery
    setTimeout(() => {
      setSentMessages(prev => prev.map(msg =>
        msg.id === message.id ? { ...msg, status: 'delivered' } : msg
      ));
    }, 2000);
  };

  // Like a community post
  const likePost = (id: string) => {
    setCommunityPosts(prev => prev.map(post =>
      post.id === id ? { ...post, likes: post.likes + 1 } : post
    ));
  };

  // Report a community post
  const reportPost = (id: string) => {
    setCommunityPosts(prev => prev.map(post =>
      post.id === id ? { ...post, reports: post.reports + 1 } : post
    ));
  };

  // Get severity emoji
  const getSeverityEmoji = (severity: string) => {
    switch (severity) {
      case 'extreme': return '🚨';
      case 'severe': return '⚠️';
      case 'warning': return '📢';
      default: return 'ℹ️';
    }
  };

  // Format time ago
  const formatTimeAgo = (timestamp: number) => {
    const minutes = Math.floor((Date.now() - timestamp) / 60000);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  return (
    <CommunicationContainer>
      <h2 style={{ marginBottom: '24px', color: '#F7F7FA' }}>
        📡 Communication Hub
      </h2>
      
      <TabBar>
        <Tab 
          isActive={activeTab === 'broadcasts'} 
          onClick={() => setActiveTab('broadcasts')}
        >
          📻 Emergency Broadcasts
        </Tab>
        <Tab 
          isActive={activeTab === 'community'} 
          onClick={() => setActiveTab('community')}
        >
          💬 Community Feed
        </Tab>
        <Tab 
          isActive={activeTab === 'messages'} 
          onClick={() => setActiveTab('messages')}
        >
          📧 Emergency Messages
        </Tab>
      </TabBar>

      {activeTab === 'broadcasts' && (
        <BroadcastList>
          {broadcasts.map(broadcast => (
            <BroadcastItem 
              key={broadcast.id}
              severity={broadcast.severity}
              isRead={broadcast.isRead}
            >
              <BroadcastHeader>
                <BroadcastInfo>
                  <div className="type">{broadcast.type} Alert</div>
                  <div className="title">
                    {getSeverityEmoji(broadcast.severity)} {broadcast.title}
                  </div>
                  <div className="issuer">
                    Issued by: {broadcast.issuedBy} • {formatTimeAgo(broadcast.issuedAt)}
                  </div>
                </BroadcastInfo>
                <SeverityBadge severity={broadcast.severity}>
                  {broadcast.severity}
                </SeverityBadge>
              </BroadcastHeader>
              
              <BroadcastContent>
                <div className="message">{broadcast.message}</div>
                
                {broadcast.affectedAreas.length > 0 && (
                  <div className="areas">
                    <strong>Affected areas:</strong> {broadcast.affectedAreas.join(', ')}
                  </div>
                )}
                
                {broadcast.actions.length > 0 && (
                  <div style={{ marginTop: '8px' }}>
                    <strong style={{ color: '#F7F7FA', fontSize: '0.9rem' }}>
                      Recommended actions:
                    </strong>
                    <ul style={{ margin: '4px 0 0 16px', color: '#D0CFD5' }}>
                      {broadcast.actions.map(action => (
                        <li key={action} style={{ fontSize: '0.85rem' }}>
                          {action}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </BroadcastContent>
              
              <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                {!broadcast.isRead && (
                  <Button
                    variant="outline"
                    onClick={() => markBroadcastAsRead(broadcast.id)}
                  >
                    ✓ Mark Read
                  </Button>
                )}
                <Button variant="outline">🔗 Share</Button>
              </div>
            </BroadcastItem>
          ))}
        </BroadcastList>
      )}

      {activeTab === 'community' && (
        <CommunityFeed>
          {communityPosts.map(post => (
            <PostItem 
              key={post.id}
              priority={post.priority}
              verified={post.verified}
            >
              <PostHeader>
                <PostAuthor>
                  <div className="name">{post.author}</div>
                  <div className="time">
                    {formatTimeAgo(post.timestamp)}
                    {post.location && ` • 📍 ${post.location}`}
                  </div>
                </PostAuthor>
                <PostBadges>
                  {post.verified && <VerifiedBadge>✓ Verified</VerifiedBadge>}
                  <CategoryBadge category={post.category}>
                    {post.category}
                  </CategoryBadge>
                </PostBadges>
              </PostHeader>
              
              <PostContent>{post.content}</PostContent>
              
              <PostActions>
                <button onClick={() => likePost(post.id)}>
                  👍 {post.likes}
                </button>
                <button>💬 Reply</button>
                <button onClick={() => reportPost(post.id)}>
                  🚩 Report ({post.reports})
                </button>
                <button>🔗 Share</button>
              </PostActions>
            </PostItem>
          ))}
        </CommunityFeed>
      )}

      {activeTab === 'messages' && (
        <div>
          <MessageComposer>
            <h4 style={{ color: '#F7F7FA', marginBottom: '16px' }}>
              📧 Send Emergency Message
            </h4>
            <MessageForm>
              <FormSelect
                value={newMessage.recipient}
                onChange={(e) => setNewMessage(prev => ({ 
                  ...prev, 
                  recipient: e.target.value as any 
                }))}
              >
                <option value="family">Family Members</option>
                <option value="emergency-services">Emergency Services</option>
                <option value="community">Community Network</option>
              </FormSelect>
              
              <FormSelect
                value={newMessage.priority}
                onChange={(e) => setNewMessage(prev => ({ 
                  ...prev, 
                  priority: e.target.value as any 
                }))}
              >
                <option value="normal">Normal Priority</option>
                <option value="high">High Priority</option>
                <option value="urgent">Urgent</option>
              </FormSelect>
              
              <FormInput
                type="text"
                placeholder="Subject"
                value={newMessage.subject}
                onChange={(e) => setNewMessage(prev => ({ 
                  ...prev, 
                  subject: e.target.value 
                }))}
              />
              
              <FormTextArea
                placeholder="Message content..."
                value={newMessage.content}
                onChange={(e) => setNewMessage(prev => ({ 
                  ...prev, 
                  content: e.target.value 
                }))}
              />
              
              <Button
                variant="primary"
                onClick={sendMessage}
                disabled={!newMessage.subject || !newMessage.content}
              >
                📧 Send Message
              </Button>
            </MessageForm>
          </MessageComposer>
          
          {sentMessages.length > 0 && (
            <div>
              <h4 style={{ color: '#F7F7FA', marginBottom: '16px' }}>
                📤 Sent Messages
              </h4>
              {sentMessages.map(message => (
                <div 
                  key={message.id}
                  style={{
                    background: '#23262C',
                    border: '1px solid #2F3339',
                    borderRadius: '8px',
                    padding: '12px',
                    marginBottom: '8px'
                  }}
                >
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    marginBottom: '8px' 
                  }}>
                    <span style={{ color: '#F7F7FA', fontWeight: '600' }}>
                      {message.subject}
                    </span>
                    <span style={{ 
                      color: message.status === 'delivered' ? '#10B981' : '#F59E0B',
                      fontSize: '0.8rem'
                    }}>
                      {message.status}
                    </span>
                  </div>
                  <div style={{ color: '#D0CFD5', fontSize: '0.9rem' }}>
                    To: {message.recipient} • {formatTimeAgo(message.sentAt!)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </CommunicationContainer>
  );
};

export default CommunicationHub;