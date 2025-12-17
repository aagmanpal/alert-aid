import React from 'react';
import styled from 'styled-components';
import { AlertTriangle, TrendingUp, Activity, BarChart3 } from 'lucide-react';
import { productionColors, productionCard } from '../styles/production-ui-system';

const PredictionsContainer = styled.div`
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

const ContentGrid = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  gap: 24px;
`;

const Card = styled.div`
  ${productionCard}
  padding: 32px;
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
`;

const CardIcon = styled.div<{ color: string }>`
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: ${({ color }) => `linear-gradient(135deg, ${color}20, ${color}10)`};
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${({ color }) => color};
`;

const CardTitle = styled.h2`
  font-size: 24px;
  font-weight: 700;
  color: ${productionColors.text.primary};
`;

const CardContent = styled.div`
  color: ${productionColors.text.secondary};
  line-height: 1.8;
`;

const PredictionsPage: React.FC = () => {
  return (
    <PredictionsContainer>
      <PageHeader>
        <PageTitle>Disaster Predictions</PageTitle>
        <PageDescription>
          AI-powered predictions and forecasts for potential disasters in your area.
          Our machine learning models analyze weather patterns, historical data, and real-time conditions.
        </PageDescription>
      </PageHeader>

      <ContentGrid>
        <Card>
          <CardHeader>
            <CardIcon color={productionColors.brand.primary}>
              <TrendingUp size={24} />
            </CardIcon>
            <CardTitle>7-Day Risk Forecast</CardTitle>
          </CardHeader>
          <CardContent>
            <p>View detailed risk predictions for the next 7 days based on weather patterns, 
            historical incident data, and current environmental conditions.</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardIcon color={productionColors.status.warning}>
              <Activity size={24} />
            </CardIcon>
            <CardTitle>Real-Time Risk Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <p>Our AI models continuously analyze current conditions and update risk scores 
            in real-time to keep you informed of changing situations.</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardIcon color={productionColors.status.info}>
              <BarChart3 size={24} />
            </CardIcon>
            <CardTitle>Historical Trends</CardTitle>
          </CardHeader>
          <CardContent>
            <p>Access historical data and trends to understand long-term patterns and 
            seasonal variations in disaster risk for your location.</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardIcon color={productionColors.status.error}>
              <AlertTriangle size={24} />
            </CardIcon>
            <CardTitle>Prediction Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <p>Our models achieve 98% accuracy in disaster prediction through continuous 
            learning and validation against real-world outcomes.</p>
          </CardContent>
        </Card>
      </ContentGrid>
    </PredictionsContainer>
  );
};

export default PredictionsPage;
