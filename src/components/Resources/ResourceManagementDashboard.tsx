import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { Card, Button } from '../../styles/components';

// Types for resource management
interface SupplyItem {
  id: string;
  name: string;
  category: 'food' | 'water' | 'medical' | 'tools' | 'communication' | 'shelter';
  currentStock: number;
  recommendedStock: number;
  unit: string;
  expiryDate?: number;
  priority: 'critical' | 'important' | 'nice-to-have';
  lastUpdated: number;
}

interface InfrastructureStatus {
  service: 'power' | 'water' | 'gas' | 'internet' | 'cellular' | 'roads' | 'bridges';
  status: 'operational' | 'limited' | 'outage' | 'unknown';
  affectedAreas: string[];
  estimatedRepair?: number;
  reportedAt: number;
  source: string;
}

interface CommunityResource {
  id: string;
  type: 'shelter' | 'food-bank' | 'medical' | 'transportation' | 'supplies' | 'volunteer';
  name: string;
  location: string;
  contact: string;
  availability: 'available' | 'limited' | 'full' | 'closed';
  description: string;
  distance: number;
  reportedBy: string;
  verifiedAt: number;
}

// Styled components
const ResourceContainer = styled(Card)`
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

const SupplyGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: ${({ theme }) => theme.spacing.md};
`;

const SupplyCard = styled.div<{ priority: string; stockLevel: number }>`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 2px solid ${({ theme, priority, stockLevel }) => 
    stockLevel < 0.2 ? theme.colors.danger[400] :
    stockLevel < 0.5 && priority === 'critical' ? theme.colors.warning[400] :
    theme.colors.surface.border
  };
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
`;

const SupplyHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const SupplyInfo = styled.div`
  flex: 1;
  
  .name {
    color: ${({ theme }) => theme.colors.text.primary};
    font-weight: 600;
    margin-bottom: 4px;
  }
  
  .category {
    color: ${({ theme }) => theme.colors.text.caption};
    font-size: 0.8rem;
    text-transform: capitalize;
  }
`;

const PriorityBadge = styled.span<{ priority: string }>`
  padding: 4px 8px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: 0.75rem;
  font-weight: 600;
  
  background: ${({ theme, priority }) => 
    priority === 'critical' ? theme.colors.danger[600] :
    priority === 'important' ? theme.colors.warning[600] :
    theme.colors.surface.border
  };
  
  color: ${({ theme, priority }) => 
    priority === 'critical' || priority === 'important' ? 
    theme.colors.text.inverse : theme.colors.text.primary
  };
`;

const StockMeter = styled.div`
  margin: ${({ theme }) => theme.spacing.sm} 0;
`;

const StockBar = styled.div<{ level: number }>`
  width: 100%;
  height: 8px;
  background: ${({ theme }) => theme.colors.surface.hover};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  overflow: hidden;
  
  &::after {
    content: '';
    display: block;
    width: ${({ level }) => Math.max(0, Math.min(100, level * 100))}%;
    height: 100%;
    background: ${({ theme, level }) => 
      level < 0.2 ? theme.colors.danger[600] :
      level < 0.5 ? theme.colors.warning[600] :
      theme.colors.success[600]
    };
    transition: width 0.3s ease;
  }
`;

const StockText = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-top: 4px;
`;

const InfrastructureGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: ${({ theme }) => theme.spacing.md};
`;

const InfrastructureCard = styled.div<{ status: string }>`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 2px solid ${({ theme, status }) => 
    status === 'operational' ? theme.colors.success[400] :
    status === 'limited' ? theme.colors.warning[400] :
    status === 'outage' ? theme.colors.danger[400] :
    theme.colors.surface.border
  };
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
`;

const StatusIcon = styled.div<{ status: string }>`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
  
  background: ${({ theme, status }) => 
    status === 'operational' ? theme.colors.success[100] :
    status === 'limited' ? theme.colors.warning[100] :
    status === 'outage' ? theme.colors.danger[100] :
    theme.colors.surface.hover
  };
`;

const ServiceName = styled.h4`
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
  text-transform: capitalize;
`;

const ServiceStatus = styled.div<{ status: string }>`
  font-weight: 600;
  font-size: 0.9rem;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
  
  color: ${({ theme, status }) => 
    status === 'operational' ? theme.colors.success[600] :
    status === 'limited' ? theme.colors.warning[600] :
    status === 'outage' ? theme.colors.danger[600] :
    theme.colors.text.secondary
  };
`;

const AffectedAreas = styled.div`
  font-size: 0.8rem;
  color: ${({ theme }) => theme.colors.text.caption};
`;

const CommunityResourcesList = styled.div`
  display: grid;
  gap: ${({ theme }) => theme.spacing.md};
`;

const ResourceCard = styled.div<{ availability: string }>`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 1px solid ${({ theme, availability }) => 
    availability === 'available' ? theme.colors.success[400] :
    availability === 'limited' ? theme.colors.warning[400] :
    availability === 'full' ? theme.colors.danger[400] :
    theme.colors.surface.border
  };
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
`;

const ResourceHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const ResourceInfo = styled.div`
  flex: 1;
  
  .type {
    color: ${({ theme }) => theme.colors.text.caption};
    font-size: 0.8rem;
    text-transform: capitalize;
    margin-bottom: 2px;
  }
  
  .name {
    color: ${({ theme }) => theme.colors.text.primary};
    font-weight: 600;
    margin-bottom: 4px;
  }
  
  .location {
    color: ${({ theme }) => theme.colors.text.secondary};
    font-size: 0.9rem;
  }
`;

const AvailabilityBadge = styled.span<{ availability: string }>`
  padding: 4px 12px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  
  background: ${({ theme, availability }) => 
    availability === 'available' ? theme.colors.success[600] :
    availability === 'limited' ? theme.colors.warning[600] :
    availability === 'full' ? theme.colors.danger[600] :
    theme.colors.surface.border
  };
  
  color: ${({ theme }) => theme.colors.text.inverse};
`;

// Get service icon
const getServiceIcon = (service: string) => {
  switch (service) {
    case 'power': return '⚡';
    case 'water': return '💧';
    case 'gas': return '🔥';
    case 'internet': return '🌐';
    case 'cellular': return '📱';
    case 'roads': return '🛣️';
    case 'bridges': return '🌉';
    default: return '🔧';
  }
};

// Get category icon
const getCategoryIcon = (category: string) => {
  switch (category) {
    case 'food': return '🥫';
    case 'water': return '💧';
    case 'medical': return '🏥';
    case 'tools': return '🔧';
    case 'communication': return '📡';
    case 'shelter': return '🏠';
    default: return '📦';
  }
};

// Main component
const ResourceManagementDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'supplies' | 'infrastructure' | 'community'>('supplies');
  
  // Mock data - in production, fetch from APIs and local storage
  const [supplies, setSupplies] = useState<SupplyItem[]>([
    {
      id: '1',
      name: 'Drinking Water',
      category: 'water',
      currentStock: 15,
      recommendedStock: 42, // 3 days × 14 gallons for family of 4
      unit: 'gallons',
      priority: 'critical',
      lastUpdated: Date.now()
    },
    {
      id: '2',
      name: 'Non-Perishable Food',
      category: 'food',
      currentStock: 6,
      recommendedStock: 12, // 3 days worth
      unit: 'days',
      priority: 'critical',
      lastUpdated: Date.now()
    },
    {
      id: '3',
      name: 'First Aid Kit',
      category: 'medical',
      currentStock: 1,
      recommendedStock: 2,
      unit: 'kits',
      priority: 'critical',
      lastUpdated: Date.now()
    },
    {
      id: '4',
      name: 'Flashlights & Batteries',
      category: 'tools',
      currentStock: 3,
      recommendedStock: 6,
      unit: 'sets',
      priority: 'important',
      lastUpdated: Date.now()
    },
    {
      id: '5',
      name: 'Battery Radio',
      category: 'communication',
      currentStock: 1,
      recommendedStock: 2,
      unit: 'devices',
      priority: 'important',
      lastUpdated: Date.now()
    },
    {
      id: '6',
      name: 'Emergency Blankets',
      category: 'shelter',
      currentStock: 2,
      recommendedStock: 6,
      unit: 'blankets',
      priority: 'important',
      lastUpdated: Date.now()
    }
  ]);

  const [infrastructure] = useState<InfrastructureStatus[]>([
    {
      service: 'power',
      status: 'operational',
      affectedAreas: [],
      reportedAt: Date.now(),
      source: 'PG&E'
    },
    {
      service: 'water',
      status: 'limited',
      affectedAreas: ['Downtown', 'East Side'],
      reportedAt: Date.now() - 3600000, // 1 hour ago
      source: 'City Water Dept'
    },
    {
      service: 'cellular',
      status: 'operational',
      affectedAreas: [],
      reportedAt: Date.now(),
      source: 'Carrier Reports'
    },
    {
      service: 'internet',
      status: 'limited',
      affectedAreas: ['Rural areas'],
      reportedAt: Date.now() - 1800000, // 30 min ago
      source: 'ISP Status'
    },
    {
      service: 'roads',
      status: 'limited',
      affectedAreas: ['Highway 101', 'Bridge St'],
      estimatedRepair: Date.now() + 7200000, // 2 hours
      reportedAt: Date.now() - 900000, // 15 min ago
      source: 'Caltrans'
    },
    {
      service: 'gas',
      status: 'operational',
      affectedAreas: [],
      reportedAt: Date.now(),
      source: 'Gas Company'
    }
  ]);

  const [communityResources] = useState<CommunityResource[]>([
    {
      id: '1',
      type: 'food-bank',
      name: 'Community Food Pantry',
      location: '123 Main St',
      contact: '(555) 123-4567',
      availability: 'available',
      description: 'Free groceries and hot meals, serving families in need',
      distance: 1.2,
      reportedBy: 'City Official',
      verifiedAt: Date.now() - 3600000
    },
    {
      id: '2',
      type: 'medical',
      name: 'Pop-up Medical Clinic',
      location: 'Central Park Pavilion',
      contact: '(555) 987-6543',
      availability: 'available',
      description: 'Basic medical care, first aid, prescription refills',
      distance: 0.8,
      reportedBy: 'Red Cross Volunteer',
      verifiedAt: Date.now() - 1800000
    },
    {
      id: '3',
      type: 'supplies',
      name: 'Emergency Supply Distribution',
      location: 'City Hall Parking Lot',
      contact: '(555) 456-7890',
      availability: 'limited',
      description: 'Water, batteries, blankets - limited quantities per family',
      distance: 2.1,
      reportedBy: 'FEMA Coordinator',
      verifiedAt: Date.now() - 900000
    },
    {
      id: '4',
      type: 'transportation',
      name: 'Evacuation Bus Service',
      location: 'Transit Center',
      contact: '(555) 321-0987',
      availability: 'available',
      description: 'Free transportation to evacuation centers',
      distance: 1.5,
      reportedBy: 'Transit Authority',
      verifiedAt: Date.now() - 600000
    }
  ]);

  // Update supply stock
  const updateSupplyStock = (id: string, newStock: number) => {
    setSupplies(prev => prev.map(supply => 
      supply.id === id 
        ? { ...supply, currentStock: newStock, lastUpdated: Date.now() }
        : supply
    ));
  };

  // Save supplies to localStorage
  useEffect(() => {
    localStorage.setItem('emergencySupplies', JSON.stringify(supplies));
  }, [supplies]);

  // Load supplies from localStorage
  useEffect(() => {
    const savedSupplies = localStorage.getItem('emergencySupplies');
    if (savedSupplies) {
      setSupplies(JSON.parse(savedSupplies));
    }
  }, []);

  return (
    <ResourceContainer>
      <h2 style={{ marginBottom: '24px', color: '#F7F7FA' }}>
        📊 Resource Management
      </h2>
      
      <TabBar>
        <Tab 
          isActive={activeTab === 'supplies'} 
          onClick={() => setActiveTab('supplies')}
        >
          📦 Emergency Supplies
        </Tab>
        <Tab 
          isActive={activeTab === 'infrastructure'} 
          onClick={() => setActiveTab('infrastructure')}
        >
          🏗️ Infrastructure Status
        </Tab>
        <Tab 
          isActive={activeTab === 'community'} 
          onClick={() => setActiveTab('community')}
        >
          🤝 Community Resources
        </Tab>
      </TabBar>

      {activeTab === 'supplies' && (
        <SupplyGrid>
          {supplies.map(supply => {
            const stockLevel = supply.currentStock / supply.recommendedStock;
            return (
              <SupplyCard 
                key={supply.id}
                priority={supply.priority}
                stockLevel={stockLevel}
              >
                <SupplyHeader>
                  <SupplyInfo>
                    <div className="name">
                      {getCategoryIcon(supply.category)} {supply.name}
                    </div>
                    <div className="category">{supply.category}</div>
                  </SupplyInfo>
                  <PriorityBadge priority={supply.priority}>
                    {supply.priority}
                  </PriorityBadge>
                </SupplyHeader>
                
                <StockMeter>
                  <StockBar level={stockLevel} />
                  <StockText>
                    <span>
                      {supply.currentStock} / {supply.recommendedStock} {supply.unit}
                    </span>
                    <span>{Math.round(stockLevel * 100)}%</span>
                  </StockText>
                </StockMeter>
                
                <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                  <Button
                    variant="outline"
                    onClick={() => updateSupplyStock(supply.id, supply.currentStock - 1)}
                    disabled={supply.currentStock <= 0}
                    style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                  >
                    -1
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => updateSupplyStock(supply.id, supply.currentStock + 1)}
                    style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                  >
                    +1
                  </Button>
                  {stockLevel < 0.5 && (
                    <Button
                      variant="primary"
                      style={{ padding: '4px 8px', fontSize: '0.8rem', marginLeft: 'auto' }}
                    >
                      🛒 Order
                    </Button>
                  )}
                </div>
              </SupplyCard>
            );
          })}
        </SupplyGrid>
      )}

      {activeTab === 'infrastructure' && (
        <InfrastructureGrid>
          {infrastructure.map(item => (
            <InfrastructureCard key={item.service} status={item.status}>
              <StatusIcon status={item.status}>
                {getServiceIcon(item.service)}
              </StatusIcon>
              
              <ServiceName>{item.service}</ServiceName>
              
              <ServiceStatus status={item.status}>
                {item.status.toUpperCase()}
              </ServiceStatus>
              
              {item.affectedAreas.length > 0 && (
                <AffectedAreas>
                  <strong>Affected areas:</strong><br />
                  {item.affectedAreas.join(', ')}
                </AffectedAreas>
              )}
              
              {item.estimatedRepair && (
                <div style={{ fontSize: '0.8rem', color: '#D0CFD5', marginTop: '8px' }}>
                  Est. repair: {new Date(item.estimatedRepair).toLocaleTimeString()}
                </div>
              )}
              
              <div style={{ fontSize: '0.75rem', color: '#9CA3AF', marginTop: '8px' }}>
                Source: {item.source}<br />
                Updated: {new Date(item.reportedAt).toLocaleTimeString()}
              </div>
            </InfrastructureCard>
          ))}
        </InfrastructureGrid>
      )}

      {activeTab === 'community' && (
        <CommunityResourcesList>
          {communityResources.map(resource => (
            <ResourceCard key={resource.id} availability={resource.availability}>
              <ResourceHeader>
                <ResourceInfo>
                  <div className="type">{resource.type.replace('-', ' ')}</div>
                  <div className="name">{resource.name}</div>
                  <div className="location">📍 {resource.location} ({resource.distance} km)</div>
                </ResourceInfo>
                <AvailabilityBadge availability={resource.availability}>
                  {resource.availability}
                </AvailabilityBadge>
              </ResourceHeader>
              
              <p style={{ color: '#D0CFD5', fontSize: '0.9rem', margin: '8px 0' }}>
                {resource.description}
              </p>
              
              <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                <Button
                  variant="primary"
                  onClick={() => window.open(`https://maps.google.com/maps/search/${encodeURIComponent(resource.location)}`)}
                >
                  🗺️ Directions
                </Button>
                <Button
                  variant="outline"
                  onClick={() => window.open(`tel:${resource.contact}`)}
                >
                  📞 Call
                </Button>
              </div>
              
              <div style={{ fontSize: '0.75rem', color: '#9CA3AF', marginTop: '8px' }}>
                Reported by: {resource.reportedBy}<br />
                Verified: {new Date(resource.verifiedAt).toLocaleTimeString()}
              </div>
            </ResourceCard>
          ))}
        </CommunityResourcesList>
      )}
    </ResourceContainer>
  );
};

export default ResourceManagementDashboard;