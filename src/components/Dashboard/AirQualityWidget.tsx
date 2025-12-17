import React from 'react';
import styled from 'styled-components';
import { Wind } from 'lucide-react';
import { AQIData } from '../../services/airQualityService';

interface AirQualityWidgetProps {
  aqiData: AQIData | null;
  loading?: boolean;
}

const AirQualityWidget: React.FC<AirQualityWidgetProps> = ({ aqiData, loading }) => {
  if (loading) {
    return (
      <Widget>
        <Header>
          <Icon><Wind size={20} /></Icon>
          <Title>Air Quality</Title>
        </Header>
        <LoadingText>Loading AQI data...</LoadingText>
      </Widget>
    );
  }

  if (!aqiData) {
    return (
      <Widget>
        <Header>
          <Icon><Wind size={20} /></Icon>
          <Title>Air Quality</Title>
        </Header>
        <ErrorText>AQI data unavailable</ErrorText>
      </Widget>
    );
  }

  const getAQIGradient = (aqi: number) => {
    const gradients: Record<number, string> = {
      1: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
      2: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
      3: 'linear-gradient(135deg, #fb923c 0%, #f97316 100%)',
      4: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
      5: 'linear-gradient(135deg, #9333ea 0%, #7e22ce 100%)',
    };
    return gradients[aqi] || gradients[3];
  };

  return (
    <Widget>
      <Header>
        <Icon><Wind size={20} /></Icon>
        <Title>Air Quality Index</Title>
        {!aqiData.is_real && <Badge>Simulated</Badge>}
      </Header>

      <AQIBadge gradient={getAQIGradient(aqiData.aqi)}>
        <AQINumber>{aqiData.aqi}</AQINumber>
        <AQILevel>{aqiData.level}</AQILevel>
        <AQIScale>Scale: 1 (Best) - 5 (Worst)</AQIScale>
      </AQIBadge>

      <Description>{aqiData.description}</Description>

      <PollutantsGrid>
        <Pollutant>
          <PollutantName>PM2.5</PollutantName>
          <PollutantValue>{aqiData.components.pm2_5} µg/m³</PollutantValue>
        </Pollutant>
        <Pollutant>
          <PollutantName>PM10</PollutantName>
          <PollutantValue>{aqiData.components.pm10} µg/m³</PollutantValue>
        </Pollutant>
        <Pollutant>
          <PollutantName>NO₂</PollutantName>
          <PollutantValue>{aqiData.components.no2} µg/m³</PollutantValue>
        </Pollutant>
        <Pollutant>
          <PollutantName>O₃</PollutantName>
          <PollutantValue>{aqiData.components.o3} µg/m³</PollutantValue>
        </Pollutant>
      </PollutantsGrid>

      {aqiData.aqi >= 3 && (
        <WarningBox severity={aqiData.aqi}>
          <WarningTitle>⚠️ Health Advisory</WarningTitle>
          <WarningText>
            {aqiData.aqi >= 4
              ? 'Avoid outdoor activities. Wear a mask if you must go outside.'
              : 'Sensitive groups should limit outdoor activities.'}
          </WarningText>
        </WarningBox>
      )}
    </Widget>
  );
};

// Styled Components
const Widget = styled.div`
  background: linear-gradient(135deg, rgba(10, 10, 10, 0.95) 0%, rgba(20, 20, 20, 0.95) 100%);
  border: 1px solid rgba(220, 38, 38, 0.3);
  border-radius: 16px;
  padding: 24px;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(220, 38, 38, 0.2);
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
`;

const Icon = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: rgba(220, 38, 38, 0.2);
  border-radius: 10px;
  color: #ef4444;
`;

const Title = styled.h3`
  color: #fff;
  font-size: 18px;
  font-weight: 700;
  margin: 0;
  flex: 1;
`;

const Badge = styled.span`
  padding: 4px 10px;
  background: rgba(251, 191, 36, 0.2);
  border: 1px solid rgba(251, 191, 36, 0.4);
  border-radius: 12px;
  color: #fbbf24;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const AQIBadge = styled.div<{ gradient: string }>`
  background: ${props => props.gradient};
  border-radius: 16px;
  padding: 24px;
  text-align: center;
  margin-bottom: 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
`;

const AQINumber = styled.div`
  color: #fff;
  font-size: 48px;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 8px;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
`;

const AQILevel = styled.div`
  color: rgba(255, 255, 255, 0.95);
  font-size: 16px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 8px;
`;

const AQIScale = styled.div`
  color: rgba(255, 255, 255, 0.7);
  font-size: 12px;
  font-weight: 500;
  font-style: italic;
`;

const Description = styled.p`
  color: rgba(255, 255, 255, 0.8);
  font-size: 14px;
  line-height: 1.6;
  margin-bottom: 20px;
  text-align: center;
`;

const PollutantsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-bottom: 16px;
`;

const Pollutant = styled.div`
  background: rgba(10, 10, 10, 0.6);
  border: 1px solid rgba(220, 38, 38, 0.2);
  border-radius: 10px;
  padding: 12px;
  transition: all 0.3s ease;

  &:hover {
    border-color: rgba(220, 38, 38, 0.4);
    background: rgba(20, 20, 20, 0.8);
  }
`;

const PollutantName = styled.div`
  color: rgba(255, 255, 255, 0.6);
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const PollutantValue = styled.div`
  color: #fff;
  font-size: 16px;
  font-weight: 700;
`;

const WarningBox = styled.div<{ severity: number }>`
  background: ${props => 
    props.severity >= 4
      ? 'rgba(239, 68, 68, 0.15)'
      : 'rgba(251, 146, 60, 0.15)'
  };
  border: 1px solid ${props => 
    props.severity >= 4
      ? 'rgba(239, 68, 68, 0.4)'
      : 'rgba(251, 146, 60, 0.4)'
  };
  border-radius: 10px;
  padding: 16px;
  margin-top: 16px;
`;

const WarningTitle = styled.div`
  color: #fff;
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 6px;
`;

const WarningText = styled.div`
  color: rgba(255, 255, 255, 0.9);
  font-size: 13px;
  line-height: 1.5;
`;

const LoadingText = styled.div`
  color: rgba(255, 255, 255, 0.6);
  font-size: 14px;
  text-align: center;
  padding: 20px;
`;

const ErrorText = styled.div`
  color: rgba(239, 68, 68, 0.8);
  font-size: 14px;
  text-align: center;
  padding: 20px;
`;

export default AirQualityWidget;
