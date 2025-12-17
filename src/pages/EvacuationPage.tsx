import React from 'react';
import styled from 'styled-components';
import EvacuationSafetyModule from '../components/Safety/EvacuationSafetyModule';
import { productionColors, productionCard } from '../styles/production-ui-system';

const EvacuationContainer = styled.div`
  min-height: 100vh;
  padding: 88px 24px 24px;
  background: ${productionColors.background.primary};
  color: ${productionColors.text.primary};
`;

const PageHeader = styled.div`
  max-width: 1200px;
  margin: 0 auto 48px;
`;

const PageTitle = styled.h1`
  font-size: 48px;
  font-weight: 800;
  background: linear-gradient(135deg, ${productionColors.brand.primary}, ${productionColors.brand.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 16px;
`;

const PageDescription = styled.p`
  font-size: 18px;
  color: ${productionColors.text.secondary};
  max-width: 800px;
`;

const EvacuationContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  gap: 24px;
`;

const EvacuationCard = styled.div`
  ${productionCard}
  padding: 32px;
`;

const EvacuationPage: React.FC = () => {
  return (
    <EvacuationContainer>
      <PageHeader>
        <PageTitle>Evacuation Routes</PageTitle>
        <PageDescription>
          Find the safest evacuation routes from your current location in case of emergency.
          Get real-time updates on route conditions and shelter locations.
        </PageDescription>
      </PageHeader>

      <EvacuationContent>
        <EvacuationCard>
          <EvacuationSafetyModule />
        </EvacuationCard>
      </EvacuationContent>
    </EvacuationContainer>
  );
};

export default EvacuationPage;
