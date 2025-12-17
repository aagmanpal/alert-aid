import React, { useMemo } from 'react';
import styled from 'styled-components';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Calendar, TrendingUp } from 'lucide-react';
import { Card, Heading, Text, Flex } from '../../styles/components';
import { ForecastData } from '../../types';
import { TemperatureConverter } from '../../utils/temperatureConverter';

const ForecastContainer = styled(Card)`
  height: 280px;
  display: flex;
  flex-direction: column;
`;

const ForecastHeader = styled(Flex)`
  margin-bottom: ${({ theme }) => theme.spacing[4]};
  padding-bottom: ${({ theme }) => theme.spacing[3]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const ChartContainer = styled.div`
  flex: 1;
  min-height: 180px;
  position: relative;
`;

const ForecastSummary = styled.div`
  margin-top: ${({ theme }) => theme.spacing[3]};
  padding-top: ${({ theme }) => theme.spacing[3]};
  border-top: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const SummaryItem = styled(Flex)`
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${({ theme }) => theme.spacing[1]};
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const CustomTooltip = styled.div`
  background: ${({ theme }) => theme.colors.surface.default};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing[3]};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.primary};
  box-shadow: ${({ theme }) => theme.shadows.lg};
`;

interface SevenDayForecastProps {
  forecast?: ForecastData[];
}

const SevenDayForecast: React.FC<SevenDayForecastProps> = React.memo(({ 
  forecast 
}) => {
  // React Hooks must be called before any early returns
  const getBarColor = useMemo(() => (riskScore: number) => {
    if (riskScore >= 8) return '#dc2626'; // Critical - Red
    if (riskScore >= 6.5) return '#f59e0b'; // High - Orange
    if (riskScore >= 4) return '#3b82f6'; // Medium - Blue
    return '#22c55e'; // Low - Green
  }, []);

  // If no forecast data provided, show loading state
  if (!forecast || forecast.length === 0) {
    return (
      <ForecastContainer>
        <ForecastHeader justify="between" align="center">
          <Flex align="center" gap="8px">
            <Calendar size={16} />
            <Heading level={5} weight="semibold">7 Day Forecast</Heading>
          </Flex>
        </ForecastHeader>
        <Flex justify="center" align="center" style={{ flex: 1 }}>
          <Text size="sm" color="secondary">Loading forecast data...</Text>
        </Flex>
      </ForecastContainer>
    );
  }

  const formatTooltip = (value: any, name: string, props: any) => {
    if (name === 'riskScore') {
      const { payload } = props;
      return [
        <CustomTooltip key="tooltip">
          <div><strong>{payload.day} Forecast</strong></div>
          <div>Risk Score: {value}</div>
          <div>Precipitation: {payload.precipitation}mm</div>
          <div>Temperature: {TemperatureConverter.formatForDisplay(payload.temperature, 'C')}</div>
          <div>Wind: {payload.windSpeed} km/h</div>
        </CustomTooltip>
      ];
    }
    return null;
  };

  const averageRisk = forecast.reduce((sum, day) => sum + day.riskScore, 0) / forecast.length;
  const maxRisk = Math.max(...forecast.map(day => day.riskScore));
  const trendDirection = forecast[forecast.length - 1].riskScore > forecast[0].riskScore ? 'up' : 'down';

  return (
    <ForecastContainer>
      <ForecastHeader justify="between" align="center">
        <Flex align="center" gap="8px">
          <Calendar size={16} />
          <Heading level={5} weight="semibold">7 Day Forecast</Heading>
        </Flex>
        <Flex align="center" gap="4px">
          <TrendingUp 
            size={12} 
            style={{ 
              transform: trendDirection === 'down' ? 'rotate(180deg)' : 'none',
              color: trendDirection === 'up' ? '#dc2626' : '#22c55e'
            }} 
          />
          <Text size="xs" color="secondary">
            {trendDirection === 'up' ? 'Increasing' : 'Decreasing'}
          </Text>
        </Flex>
      </ForecastHeader>
      
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={forecast} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
            <XAxis 
              dataKey="day" 
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: '#64748b' }}
            />
            <YAxis 
              domain={[0, 10]}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: '#64748b' }}
              width={25}
            />
            <Tooltip 
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return formatTooltip(data.riskScore, 'riskScore', { payload: data });
                }
                return null;
              }}
            />
            <Bar 
              dataKey="riskScore" 
              radius={[2, 2, 0, 0]}
            >
              {forecast.map((entry, index) => (
                <Cell key={index} fill={getBarColor(entry.riskScore)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
      
      <ForecastSummary>
        <SummaryItem>
          <Text size="xs" color="secondary">Average Risk</Text>
          <Text size="xs" weight="medium">{averageRisk.toFixed(1)}</Text>
        </SummaryItem>
        <SummaryItem>
          <Text size="xs" color="secondary">Peak Risk</Text>
          <Text size="xs" weight="medium" style={{ color: getBarColor(maxRisk) }}>
            {maxRisk.toFixed(1)}
          </Text>
        </SummaryItem>
      </ForecastSummary>
    </ForecastContainer>
  );
});

export default SevenDayForecast;