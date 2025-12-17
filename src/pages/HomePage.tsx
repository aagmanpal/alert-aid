import React from 'react';
import styled from 'styled-components';
import { TrendingUp, Shield, Activity } from 'lucide-react';
import { productionColors, productionCard } from '../styles/production-ui-system';

const HomeContainer = styled.div`
  min-height: 100vh;
  padding: 88px 24px 24px;
  background: ${productionColors.background.primary};
  color: ${productionColors.text.primary};
`;

const HeroSection = styled.div`
  max-width: 1200px;
  margin: 0 auto 48px;
  text-align: center;
`;

const Title = styled.h1`
  font-size: 56px;
  font-weight: 800;
  background: linear-gradient(135deg, ${productionColors.brand.primary}, ${productionColors.brand.secondary});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 24px;
`;

const Subtitle = styled.p`
  font-size: 24px;
  color: ${productionColors.text.secondary};
  max-width: 800px;
  margin: 0 auto;
  line-height: 1.6;
`;

const FeaturesGrid = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
`;

const FeatureCard = styled.div`
  ${productionCard}
  padding: 32px;
  text-align: center;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-8px);
    border-color: ${productionColors.brand.primary};
  }
`;

const FeatureIcon = styled.div`
  width: 80px;
  height: 80px;
  margin: 0 auto 24px;
  border-radius: 50%;
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(249, 115, 22, 0.2));
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${productionColors.brand.primary};
`;

const FeatureTitle = styled.h3`
  font-size: 24px;
  font-weight: 700;
  color: ${productionColors.text.primary};
  margin-bottom: 16px;
`;

const FeatureDescription = styled.p`
  font-size: 16px;
  color: ${productionColors.text.secondary};
  line-height: 1.6;
`;

const CTAButton = styled.button`
  margin: 48px auto 0;
  display: block;
  padding: 16px 48px;
  font-size: 18px;
  font-weight: 700;
  color: white;
  background: ${productionColors.gradients.brand};
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 8px 24px rgba(239, 68, 68, 0.3);
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(239, 68, 68, 0.4);
  }
`;

const HomePage: React.FC = () => {
  return (
    <HomeContainer>
      <HeroSection>
        <Title>Alert Aid</Title>
        <Subtitle>
          Advanced disaster prediction and response system powered by AI. 
          Stay prepared, stay safe with real-time alerts and intelligent forecasting.
        </Subtitle>
      </HeroSection>

      <FeaturesGrid>
        <FeatureCard>
          <FeatureIcon>
            <TrendingUp size={40} />
          </FeatureIcon>
          <FeatureTitle>AI-Powered Predictions</FeatureTitle>
          <FeatureDescription>
            Machine learning models analyze weather patterns and historical data 
            to predict disasters with 98% accuracy.
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>
            <Activity size={40} />
          </FeatureIcon>
          <FeatureTitle>Real-Time Monitoring</FeatureTitle>
          <FeatureDescription>
            Live weather tracking, risk assessment, and instant notifications 
            keep you informed 24/7.
          </FeatureDescription>
        </FeatureCard>

        <FeatureCard>
          <FeatureIcon>
            <Shield size={40} />
          </FeatureIcon>
          <FeatureTitle>Emergency Response</FeatureTitle>
          <FeatureDescription>
            Integrated evacuation routes, emergency contacts, and safety 
            protocols at your fingertips.
          </FeatureDescription>
        </FeatureCard>
      </FeaturesGrid>

      <CTAButton onClick={() => window.location.href = '/dashboard'}>
        Go to Dashboard
      </CTAButton>
    </HomeContainer>
  );
};

export default HomePage;
