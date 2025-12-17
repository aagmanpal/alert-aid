import React from 'react';
import styled from 'styled-components';
import { spacing, gridSpacing, breakpoints } from '../../styles/spacing';
import CurrentAlerts from './CurrentAlerts';
import MLPredictionAccuracy from './MLPredictionAccuracy';
import ActionButtons from './ActionButtons';
import Central3DLandscape from '../ThreeD/Central3DLandscape';
import SevenDayForecast from './SevenDayForecast';
import HistoricalTrends from './HistoricalTrends';
import GeolocationManager from '../Location/GeolocationManager';
import EmergencyResponsePanel from '../Emergency/EmergencyResponsePanel';
import EvacuationSafetyModule from '../Safety/EvacuationSafetyModule';
import ResourceManagementDashboard from '../Resources/ResourceManagementDashboard';
import CommunicationHub from '../Communication/CommunicationHub';

const DashboardContainer = styled.main`
  min-height: 100vh;
  padding-top: 70px; /* Account for fixed header */
  background: ${({ theme }) => theme.colors.background.primary};
  background-image: 
    radial-gradient(circle at 25% 25%, ${({ theme }) => theme.colors.primary[50]}15 0%, transparent 50%),
    radial-gradient(circle at 75% 75%, ${({ theme }) => theme.colors.primary[100]}10 0%, transparent 40%);
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
`;

// 24px Base Spacing Grid - NO OVERLAPPING
const DashboardGrid = styled.div`
  display: grid;
  gap: ${gridSpacing.container}; /* 24px gaps between all sections */
  padding: ${spacing.xl}; /* 24px container padding */
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  
  /* Desktop: 3-column layout (1200px+) */
  @media (min-width: ${breakpoints.desktop}) {
    grid-template-columns: 300px 1fr 300px; /* Fixed sidebars, flexible center */
    grid-template-areas:
      "left center right"
      "emergency emergency emergency"
      "evacuation resources communication";
  }
  
  /* Tablet: 2-column layout (768px-1199px) */
  @media (min-width: ${breakpoints.tablet}) and (max-width: ${breakpoints.tabletMax}) {
    grid-template-columns: 1fr 1fr;
    grid-template-areas:
      "left center"
      "right center"
      "emergency emergency"
      "evacuation resources"
      "communication communication";
    gap: ${spacing.lg}; /* 16px on tablet */
  }
  
  /* Mobile: Single column (below 768px) */
  @media (max-width: ${breakpoints.mobile}) {
    grid-template-columns: 1fr;
    grid-template-areas:
      "center"
      "left"
      "right"
      "emergency"
      "evacuation"
      "resources"
      "communication";
    gap: ${spacing.lg}; /* 16px on mobile */
    padding: ${spacing.md}; /* 12px on mobile */
  }
`;

const LeftSidebar = styled.aside`
  grid-area: left;
  display: flex;
  flex-direction: column;
  gap: ${gridSpacing.minMargin}; /* 16px between cards */
  min-height: 0; /* Prevent flex overflow */
`;

const CenterArea = styled.section`
  grid-area: center;
  display: flex;
  flex-direction: column;
  gap: ${gridSpacing.minMargin}; /* 16px between elements */
  align-items: center;
  justify-content: flex-start;
  min-height: 0;
`;

const RightSidebar = styled.aside`
  grid-area: right;
  display: flex;
  flex-direction: column;
  gap: ${gridSpacing.minMargin}; /* 16px between cards */
  min-height: 0;
`;

// Constrained 3D visualization - exactly 350x350px
const VisualizationContainer = styled.div`
  width: 350px;
  height: 350px;
  max-width: 100%;
  margin-bottom: ${spacing.xl}; /* 24px margin below */
  
  @media (max-width: ${breakpoints.mobile}) {
    width: 280px;
    height: 280px;
    margin-bottom: ${spacing.lg}; /* 16px on mobile */
  }
`;

// Standardized card container with consistent spacing
const DashboardCard = styled.div`
  background: ${({ theme }) => theme.colors.surface.elevated};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: 12px;
  padding: ${gridSpacing.cardPadding}; /* 20px internal padding */
  margin: ${gridSpacing.minMargin}; /* 16px minimum margins */
  min-height: 200px; /* Standardized minimum height */
  box-shadow: ${({ theme }) => theme.shadows.soft};
  transition: all 0.3s ease;
  
  &:hover {
    box-shadow: ${({ theme }) => theme.shadows.medium};
    transform: translateY(-1px);
  }
  
  /* Ensure no overlapping */
  position: relative;
  z-index: 1;
`;

// Emergency section spans full width
const EmergencySection = styled.section`
  grid-area: emergency;
  display: grid;
  grid-template-columns: 1fr;
  gap: ${spacing.lg}; /* 16px gap */
`;

const EvacuationSection = styled.section`
  grid-area: evacuation;
`;

const ResourcesSection = styled.section`
  grid-area: resources;
`;

const CommunicationSection = styled.section`
  grid-area: communication;
`;

const Dashboard: React.FC = () => {
  return (
    <DashboardContainer>
      <GeolocationManager />
      
      <DashboardGrid className="dashboard-grid">
        {/* Left Sidebar - Alerts & Controls */}
        <LeftSidebar>
          <DashboardCard>
            <CurrentAlerts />
          </DashboardCard>
          
          <DashboardCard>
            <MLPredictionAccuracy />
          </DashboardCard>
          
          <DashboardCard>
            <ActionButtons />
          </DashboardCard>
        </LeftSidebar>

        {/* Center Area - 3D Visualization */}
        <CenterArea>
          <VisualizationContainer className="visualization-container">
            <Central3DLandscape />
          </VisualizationContainer>
        </CenterArea>

        {/* Right Sidebar - Forecast & Trends */}
        <RightSidebar>
          <DashboardCard>
            <SevenDayForecast />
          </DashboardCard>
          
          <DashboardCard>
            <HistoricalTrends />
          </DashboardCard>
        </RightSidebar>

        {/* Emergency Response - Full Width */}
        <EmergencySection>
          <DashboardCard>
            <EmergencyResponsePanel />
          </DashboardCard>
        </EmergencySection>

        {/* Evacuation & Safety */}
        <EvacuationSection>
          <DashboardCard>
            <EvacuationSafetyModule />
          </DashboardCard>
        </EvacuationSection>

        {/* Resource Management */}
        <ResourcesSection>
          <DashboardCard>
            <ResourceManagementDashboard />
          </DashboardCard>
        </ResourcesSection>

        {/* Communication Hub */}
        <CommunicationSection>
          <DashboardCard>
            <CommunicationHub />
          </DashboardCard>
        </CommunicationSection>
      </DashboardGrid>
    </DashboardContainer>
  );
};

export default Dashboard;