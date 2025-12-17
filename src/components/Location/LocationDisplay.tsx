import React from 'react';
import styled from 'styled-components';
import { MapPin, Settings } from 'lucide-react';
import { useGeolocation } from './GeolocationManager';
import { spacing } from '../../styles/spacing';

const LocationContainer = styled.div`
  display: flex;
  align-items: center;
  gap: ${spacing.sm};
  background: ${({ theme }) => theme.colors.surface.default};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: 8px;
  padding: ${spacing.sm} ${spacing.md};
  transition: all 0.2s ease;
  cursor: pointer;
  
  &:hover {
    background: ${({ theme }) => theme.colors.surface.hover};
    border-color: ${({ theme }) => theme.colors.border.medium};
  }
  
  @media (max-width: 768px) {
    padding: ${spacing.xs} ${spacing.sm};
  }
`;

const LocationIcon = styled.div`
  color: ${({ theme }) => theme.colors.primary[600]};
  display: flex;
  align-items: center;
`;

const LocationText = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0; /* Allow text truncation */
`;

const LocationCity = styled.span`
  color: ${({ theme }) => theme.colors.text.primary};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
  
  @media (max-width: 768px) {
    max-width: 120px;
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
  }
`;

const LocationStatus = styled.span`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  white-space: nowrap;
`;

const LocationButton = styled.button`
  background: none;
  border: none;
  padding: ${spacing.xs};
  border-radius: 4px;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.text.secondary};
  transition: all 0.2s ease;
  
  &:hover {
    background: ${({ theme }) => theme.colors.surface.hover};
    color: ${({ theme }) => theme.colors.text.primary};
  }
`;

const NoLocationContainer = styled.div`
  display: flex;
  align-items: center;
  gap: ${spacing.sm};
  background: ${({ theme }) => theme.colors.warning[50]};
  border: 1px solid ${({ theme }) => theme.colors.warning[300]};
  border-radius: 8px;
  padding: ${spacing.sm} ${spacing.md};
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${({ theme }) => theme.colors.warning[100]};
  }
`;

const NoLocationText = styled.span`
  color: ${({ theme }) => theme.colors.warning[700]};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  
  @media (max-width: 768px) {
    font-size: ${({ theme }) => theme.typography.fontSize.xs};
  }
`;

export const LocationDisplay: React.FC = () => {
  const { location, isLoading, requestLocation } = useGeolocation();

  const handleLocationClick = () => {
    if (!location) {
      requestLocation();
    }
    // TODO: In the future, this could open a location settings modal
  };

  if (isLoading) {
    return (
      <LocationContainer>
        <LocationIcon>
          <MapPin size={16} />
        </LocationIcon>
        <LocationText>
          <LocationCity>Getting location...</LocationCity>
          <LocationStatus>Please wait</LocationStatus>
        </LocationText>
      </LocationContainer>
    );
  }

  if (!location) {
    return (
      <NoLocationContainer onClick={handleLocationClick}>
        <LocationIcon>
          <MapPin size={16} />
        </LocationIcon>
        <NoLocationText>Set Location</NoLocationText>
      </NoLocationContainer>
    );
  }

  return (
    <LocationContainer onClick={handleLocationClick}>
      <LocationIcon>
        <MapPin size={16} />
      </LocationIcon>
      <LocationText>
        <LocationCity title={`${location.city}, ${location.state}, ${location.country}`}>
          {location.city}, {location.state}
        </LocationCity>
        <LocationStatus>
          {new Date(location.timestamp).toLocaleDateString()}
        </LocationStatus>
      </LocationText>
      <LocationButton onClick={(e) => {
        e.stopPropagation();
        requestLocation();
      }}>
        <Settings size={14} />
      </LocationButton>
    </LocationContainer>
  );
};