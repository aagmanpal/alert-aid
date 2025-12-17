import React from 'react';
import styled from 'styled-components';
import { CinematicDashboard } from '../components/Dashboard';

const DashboardWrapper = styled.div`
  width: 100%;
  min-height: 100vh;
  overflow-x: hidden;
  overflow-y: auto;
  
  /* Smooth scrolling on mobile */
  -webkit-overflow-scrolling: touch;
  
  /* Prevent zoom on double tap (mobile) */
  touch-action: manipulation;
  
  /* Optimize for mobile performance */
  @media (max-width: 768px) {
    will-change: scroll-position;
  }
`;

const DashboardDemo: React.FC = () => {
  return (
    <DashboardWrapper>
      <CinematicDashboard />
    </DashboardWrapper>
  );
};

export default DashboardDemo;