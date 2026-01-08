import React, { useState } from 'react';
import styled from 'styled-components';
import { Card, Button } from '../../styles/components';
import { useGeolocation } from '../Location/GeolocationManager';

// Types for evacuation system
interface EvacuationRoute {
  id: string;
  name: string;
  distance: number; // in kilometers
  estimatedTime: number; // in minutes
  difficulty: 'easy' | 'moderate' | 'difficult';
  transportMode: 'walking' | 'driving' | 'public-transport';
  waypoints: Waypoint[];
  hazards: string[];
  lastUpdated: number;
}

interface Waypoint {
  latitude: number;
  longitude: number;
  instruction: string;
  landmark?: string;
}

interface EmergencyShelter {
  id: string;
  name: string;
  address: string;
  latitude: number;
  longitude: number;
  capacity: number;
  currentOccupancy: number;
  distance: number;
  amenities: string[];
  contactInfo: string;
  acceptsPets: boolean;
  accessibility: boolean;
  status: 'open' | 'full' | 'closed' | 'unknown';
}

interface SafetyChecklist {
  category: 'earthquake' | 'flood' | 'wildfire' | 'storm' | 'general';
  items: SafetyChecklistItem[];
}

interface SafetyChecklistItem {
  id: string;
  text: string;
  isCompleted: boolean;
  priority: 'high' | 'medium' | 'low';
  timeFrame: 'immediate' | 'within-hour' | 'within-day';
}

// Styled components
const EvacuationContainer = styled(Card)`
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

const RouteCard = styled.div`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 1px solid ${({ theme }) => theme.colors.surface.border};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.md};
`;

const RouteHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const RouteInfo = styled.div`
  flex: 1;
  
  .name {
    color: ${({ theme }) => theme.colors.text.primary};
    font-weight: 600;
    margin-bottom: 4px;
  }
  
  .details {
    color: ${({ theme }) => theme.colors.text.secondary};
    font-size: 0.9rem;
    display: flex;
    gap: ${({ theme }) => theme.spacing.md};
    flex-wrap: wrap;
  }
`;

const DifficultyBadge = styled.span<{ difficulty: string }>`
  padding: 4px 8px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: 0.75rem;
  font-weight: 500;
  
  background: ${({ theme, difficulty }) => 
    difficulty === 'easy' ? theme.colors.success[100] :
    difficulty === 'moderate' ? theme.colors.warning[100] :
    theme.colors.danger[100]
  };
  
  color: ${({ theme, difficulty }) => 
    difficulty === 'easy' ? theme.colors.success[700] :
    difficulty === 'moderate' ? theme.colors.warning[700] :
    theme.colors.danger[700]
  };
`;

const ShelterGrid = styled.div`
  display: grid;
  gap: ${({ theme }) => theme.spacing.md};
`;

const ShelterCard = styled.div<{ status: string }>`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 2px solid ${({ theme, status }) => 
    status === 'open' ? theme.colors.success[400] :
    status === 'full' ? theme.colors.warning[400] :
    status === 'closed' ? theme.colors.danger[400] :
    theme.colors.surface.border
  };
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing.md};
`;

const ShelterHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const ShelterInfo = styled.div`
  .name {
    color: ${({ theme }) => theme.colors.text.primary};
    font-weight: 600;
    margin-bottom: 4px;
  }
  
  .address {
    color: ${({ theme }) => theme.colors.text.secondary};
    font-size: 0.9rem;
    margin-bottom: ${({ theme }) => theme.spacing.xs};
  }
  
  .capacity {
    color: ${({ theme }) => theme.colors.text.caption};
    font-size: 0.85rem;
  }
`;

const StatusBadge = styled.span<{ status: string }>`
  padding: 4px 12px;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  
  background: ${({ theme, status }) => 
    status === 'open' ? theme.colors.success[600] :
    status === 'full' ? theme.colors.warning[600] :
    status === 'closed' ? theme.colors.danger[600] :
    theme.colors.surface.border
  };
  
  color: ${({ theme }) => theme.colors.text.inverse};
`;

const AmenitiesList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing.xs};
  margin-top: ${({ theme }) => theme.spacing.sm};
`;

const AmenityTag = styled.span`
  padding: 2px 8px;
  background: ${({ theme }) => theme.colors.surface.hover};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: 0.75rem;
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const ChecklistContainer = styled.div`
  max-height: 400px;
  overflow-y: auto;
`;

const ChecklistCategory = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing.lg};
  
  h4 {
    color: ${({ theme }) => theme.colors.text.primary};
    margin-bottom: ${({ theme }) => theme.spacing.md};
    display: flex;
    align-items: center;
    gap: ${({ theme }) => theme.spacing.xs};
  }
`;

const ChecklistItem = styled.div<{ priority: string; isCompleted: boolean }>`
  display: flex;
  align-items: start;
  padding: ${({ theme }) => theme.spacing.md};
  margin-bottom: ${({ theme }) => theme.spacing.xs};
  background: ${({ theme, isCompleted }) => 
    isCompleted ? theme.colors.success[50] : theme.colors.surface.default
  };
  border: 1px solid ${({ theme, priority, isCompleted }) => 
    isCompleted ? theme.colors.success[300] :
    priority === 'high' ? theme.colors.danger[400] :
    priority === 'medium' ? theme.colors.warning[400] :
    theme.colors.surface.border
  };
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  opacity: ${({ isCompleted }) => isCompleted ? 0.7 : 1};
`;

const Checkbox = styled.input`
  margin-right: ${({ theme }) => theme.spacing.sm};
  margin-top: 2px;
`;

const ItemText = styled.div<{ isCompleted: boolean }>`
  flex: 1;
  color: ${({ theme, isCompleted }) => 
    isCompleted ? theme.colors.text.secondary : theme.colors.text.primary
  };
  text-decoration: ${({ isCompleted }) => isCompleted ? 'line-through' : 'none'};
  
  .timeframe {
    font-size: 0.8rem;
    color: ${({ theme }) => theme.colors.text.caption};
    margin-top: 4px;
  }
`;

const PriorityBadge = styled.span<{ priority: string }>`
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.borderRadius.xs};
  font-size: 0.7rem;
  font-weight: 600;
  margin-left: ${({ theme }) => theme.spacing.sm};
  
  background: ${({ theme, priority }) => 
    priority === 'high' ? theme.colors.danger[600] :
    priority === 'medium' ? theme.colors.warning[600] :
    theme.colors.surface.border
  };
  
  color: ${({ theme, priority }) => 
    priority === 'high' || priority === 'medium' ? 
    theme.colors.text.inverse : theme.colors.text.primary
  };
`;

// Main component
const EvacuationSafetyModule: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'routes' | 'shelters' | 'safety'>('routes');
  const { locationState } = useGeolocation();
  const userLocation = locationState.coordinates ? { lat: locationState.coordinates.latitude, lon: locationState.coordinates.longitude } : null;
  
  // Mock data - in production, fetch from APIs
  const [evacuationRoutes] = useState<EvacuationRoute[]>([
    {
      id: '1',
      name: 'Highway 101 North - Main Evacuation Route',
      distance: 15.2,
      estimatedTime: 25,
      difficulty: 'easy',
      transportMode: 'driving',
      waypoints: [],
      hazards: ['Traffic congestion expected', 'Bridge construction'],
      lastUpdated: Date.now()
    },
    {
      id: '2',
      name: 'Coastal Highway - Alternative Route',
      distance: 22.8,
      estimatedTime: 35,
      difficulty: 'moderate',
      transportMode: 'driving',
      waypoints: [],
      hazards: ['Possible flooding on low sections'],
      lastUpdated: Date.now()
    },
    {
      id: '3',
      name: 'Rail Trail - Walking/Cycling Route',
      distance: 8.5,
      estimatedTime: 90,
      difficulty: 'moderate',
      transportMode: 'walking',
      waypoints: [],
      hazards: ['Narrow path sections'],
      lastUpdated: Date.now()
    }
  ]);

  const [emergencyShelters] = useState<EmergencyShelter[]>([
    {
      id: '1',
      name: 'Central Community Center',
      address: '123 Main Street, Downtown',
      latitude: 37.7749,
      longitude: -122.4194,
      capacity: 500,
      currentOccupancy: 180,
      distance: 2.3,
      amenities: ['Food', 'Medical', 'WiFi', 'Showers', 'Childcare'],
      contactInfo: '(555) 123-4567',
      acceptsPets: true,
      accessibility: true,
      status: 'open'
    },
    {
      id: '2',
      name: 'Lincoln High School Gymnasium',
      address: '456 Oak Avenue, Westside',
      latitude: 37.7849,
      longitude: -122.4294,
      capacity: 300,
      currentOccupancy: 295,
      distance: 4.1,
      amenities: ['Food', 'Medical', 'Restrooms'],
      contactInfo: '(555) 987-6543',
      acceptsPets: false,
      accessibility: true,
      status: 'full'
    },
    {
      id: '3',
      name: 'Riverside Park Pavilion',
      address: '789 River Road, Eastside',
      latitude: 37.7649,
      longitude: -122.4094,
      capacity: 150,
      currentOccupancy: 0,
      distance: 6.7,
      amenities: ['Basic shelter', 'Restrooms', 'Water'],
      contactInfo: '(555) 456-7890',
      acceptsPets: true,
      accessibility: false,
      status: 'open'
    }
  ]);

  const [safetyChecklists, setSafetyChecklists] = useState<SafetyChecklist[]>([
    {
      category: 'earthquake',
      items: [
        { id: '1', text: 'Drop, Cover, and Hold On if shaking starts', isCompleted: false, priority: 'high', timeFrame: 'immediate' },
        { id: '2', text: 'Stay away from windows and heavy objects', isCompleted: false, priority: 'high', timeFrame: 'immediate' },
        { id: '3', text: 'Check for injuries and provide first aid', isCompleted: false, priority: 'high', timeFrame: 'immediate' },
        { id: '4', text: 'Turn off gas, water, and electricity if damaged', isCompleted: false, priority: 'medium', timeFrame: 'within-hour' },
        { id: '5', text: 'Check building for structural damage', isCompleted: false, priority: 'medium', timeFrame: 'within-hour' },
        { id: '6', text: 'Contact family members to confirm safety', isCompleted: false, priority: 'medium', timeFrame: 'within-hour' }
      ]
    },
    {
      category: 'wildfire',
      items: [
        { id: '7', text: 'Monitor emergency broadcasts for evacuation orders', isCompleted: false, priority: 'high', timeFrame: 'immediate' },
        { id: '8', text: 'Prepare go-bag with essentials', isCompleted: false, priority: 'high', timeFrame: 'immediate' },
        { id: '9', text: 'Close all windows and doors', isCompleted: false, priority: 'medium', timeFrame: 'immediate' },
        { id: '10', text: 'Move flammable items away from windows', isCompleted: false, priority: 'medium', timeFrame: 'within-hour' },
        { id: '11', text: 'Fill bathtubs and containers with water', isCompleted: false, priority: 'medium', timeFrame: 'within-hour' },
        { id: '12', text: 'Evacuate immediately if ordered', isCompleted: false, priority: 'high', timeFrame: 'immediate' }
      ]
    },
    {
      category: 'flood',
      items: [
        { id: '13', text: 'Move to higher ground immediately', isCompleted: false, priority: 'high', timeFrame: 'immediate' },
        { id: '14', text: 'Avoid walking or driving through flood water', isCompleted: false, priority: 'high', timeFrame: 'immediate' },
        { id: '15', text: 'Turn off electricity at main breaker', isCompleted: false, priority: 'high', timeFrame: 'immediate' },
        { id: '16', text: 'Gather emergency supplies', isCompleted: false, priority: 'medium', timeFrame: 'immediate' },
        { id: '17', text: 'Stay tuned to weather radio', isCompleted: false, priority: 'medium', timeFrame: 'immediate' }
      ]
    }
  ]);

  // Get current location - handled by useGeolocation hook

  // Navigate to route
  const navigateToRoute = (route: EvacuationRoute) => {
    if (userLocation) {
      const url = `https://www.google.com/maps/dir/${userLocation.lat},${userLocation.lon}/destination`;
      window.open(url, '_blank');
    }
  };

  // Navigate to shelter
  const navigateToShelter = (shelter: EmergencyShelter) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${shelter.latitude},${shelter.longitude}`;
    window.open(url, '_blank');
  };

  // Toggle checklist item
  const toggleChecklistItem = (category: string, itemId: string) => {
    setSafetyChecklists(prev => prev.map(checklist => 
      checklist.category === category
        ? {
            ...checklist,
            items: checklist.items.map(item =>
              item.id === itemId
                ? { ...item, isCompleted: !item.isCompleted }
                : item
            )
          }
        : checklist
    ));
  };

  // Get category emoji
  const getCategoryEmoji = (category: string) => {
    switch (category) {
      case 'earthquake': return '🌍';
      case 'wildfire': return '🔥';
      case 'flood': return '🌊';
      case 'storm': return '⛈️';
      default: return '📋';
    }
  };

  // Get transport mode emoji  
  const getTransportEmoji = (mode: string) => {
    switch (mode) {
      case 'driving': return '🚗';
      case 'walking': return '🚶';
      case 'public-transport': return '🚌';
      default: return '🗺️';
    }
  };

  return (
    <EvacuationContainer>
      <h2 style={{ marginBottom: '24px', color: '#F7F7FA' }}>
        🚨 Evacuation & Safety
      </h2>
      
      <TabBar>
        <Tab 
          isActive={activeTab === 'routes'} 
          onClick={() => setActiveTab('routes')}
        >
          🗺️ Evacuation Routes
        </Tab>
        <Tab 
          isActive={activeTab === 'shelters'} 
          onClick={() => setActiveTab('shelters')}
        >
          🏠 Emergency Shelters
        </Tab>
        <Tab 
          isActive={activeTab === 'safety'} 
          onClick={() => setActiveTab('safety')}
        >
          ✅ Safety Checklist
        </Tab>
      </TabBar>

      {activeTab === 'routes' && (
        <div>
          {evacuationRoutes.map(route => (
            <RouteCard key={route.id}>
              <RouteHeader>
                <RouteInfo>
                  <div className="name">{route.name}</div>
                  <div className="details">
                    <span>{getTransportEmoji(route.transportMode)} {route.transportMode}</span>
                    <span>📍 {route.distance} km</span>
                    <span>⏱️ {route.estimatedTime} min</span>
                    <DifficultyBadge difficulty={route.difficulty}>
                      {route.difficulty}
                    </DifficultyBadge>
                  </div>
                </RouteInfo>
                <Button
                  variant="primary"
                  onClick={() => navigateToRoute(route)}
                >
                  🗺️ Navigate
                </Button>
              </RouteHeader>
              {route.hazards.length > 0 && (
                <div style={{ marginTop: '12px' }}>
                  <strong style={{ color: '#F04449', fontSize: '0.9rem' }}>
                    ⚠️ Current Hazards:
                  </strong>
                  <ul style={{ margin: '4px 0 0 0', color: '#D0CFD5' }}>
                    {route.hazards.map(hazard => (
                      <li key={hazard} style={{ fontSize: '0.85rem' }}>
                        {hazard}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </RouteCard>
          ))}
        </div>
      )}

      {activeTab === 'shelters' && (
        <ShelterGrid>
          {emergencyShelters.map(shelter => (
            <ShelterCard key={shelter.id} status={shelter.status}>
              <ShelterHeader>
                <ShelterInfo>
                  <div className="name">{shelter.name}</div>
                  <div className="address">{shelter.address}</div>
                  <div className="capacity">
                    Capacity: {shelter.currentOccupancy}/{shelter.capacity} 
                    ({Math.round((shelter.capacity - shelter.currentOccupancy) / shelter.capacity * 100)}% available)
                  </div>
                </ShelterInfo>
                <StatusBadge status={shelter.status}>
                  {shelter.status}
                </StatusBadge>
              </ShelterHeader>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ color: '#D0CFD5', fontSize: '0.9rem' }}>
                  📍 {shelter.distance} km away
                </span>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {shelter.acceptsPets && <span title="Pet-friendly">🐕</span>}
                  {shelter.accessibility && <span title="Wheelchair accessible">♿</span>}
                </div>
              </div>
              
              <AmenitiesList>
                {shelter.amenities.map(amenity => (
                  <AmenityTag key={amenity}>{amenity}</AmenityTag>
                ))}
              </AmenitiesList>
              
              <div style={{ marginTop: '16px', display: 'flex', gap: '8px' }}>
                <Button
                  variant="primary"
                  onClick={() => navigateToShelter(shelter)}
                  disabled={shelter.status === 'closed'}
                >
                  🗺️ Navigate
                </Button>
                <Button
                  variant="outline"
                  onClick={() => window.open(`tel:${shelter.contactInfo}`)}
                >
                  📞 Call
                </Button>
              </div>
            </ShelterCard>
          ))}
        </ShelterGrid>
      )}

      {activeTab === 'safety' && (
        <ChecklistContainer>
          {safetyChecklists.map(checklist => (
            <ChecklistCategory key={checklist.category}>
              <h4>
                {getCategoryEmoji(checklist.category)}
                {checklist.category.charAt(0).toUpperCase() + checklist.category.slice(1)} Safety
              </h4>
              {checklist.items.map(item => (
                <ChecklistItem 
                  key={item.id}
                  priority={item.priority}
                  isCompleted={item.isCompleted}
                >
                  <Checkbox
                    type="checkbox"
                    checked={item.isCompleted}
                    onChange={() => toggleChecklistItem(checklist.category, item.id)}
                  />
                  <ItemText isCompleted={item.isCompleted}>
                    {item.text}
                    <div className="timeframe">
                      ⏰ {item.timeFrame.replace('-', ' ')}
                    </div>
                  </ItemText>
                  <PriorityBadge priority={item.priority}>
                    {item.priority}
                  </PriorityBadge>
                </ChecklistItem>
              ))}
            </ChecklistCategory>
          ))}
        </ChecklistContainer>
      )}
    </EvacuationContainer>
  );
};

export default EvacuationSafetyModule;