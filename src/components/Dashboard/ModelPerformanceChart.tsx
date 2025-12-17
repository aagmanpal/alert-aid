import React from 'react';
import styled from 'styled-components';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Brain, TrendingUp } from 'lucide-react';
import { Card, Heading, Text, Flex } from '../../styles/components';
import { ModelPerformance } from '../../types';

const PerformanceContainer = styled(Card)`
  height: 280px;
  display: flex;
  flex-direction: column;
`;

const PerformanceHeader = styled(Flex)`
  margin-bottom: ${({ theme }) => theme.spacing.md};
  padding-bottom: ${({ theme }) => theme.spacing.sm};
  border-bottom: 1px solid ${({ theme }) => theme.colors.surface};
`;

const ChartSection = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  position: relative;
`;

const ChartContainer = styled.div`
  flex: 1;
  height: 120px;
  position: relative;
`;

const CenterValue = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  pointer-events: none;
`;

const MainMetric = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.success};
  line-height: 1;
`;

const MetricLabel = styled(Text)`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  margin-top: 2px;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing.sm};
  margin-top: ${({ theme }) => theme.spacing.sm};
  padding-top: ${({ theme }) => theme.spacing.sm};
  border-top: 1px solid ${({ theme }) => theme.colors.surface};
`;

const MetricItem = styled.div`
  text-align: center;
`;

const MetricValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  line-height: 1;
`;

const MetricName = styled(Text)`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  margin-top: 2px;
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing.xs};
  color: ${({ theme }) => theme.colors.success};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  
  svg {
    width: 12px;
    height: 12px;
  }
`;

interface ModelPerformanceChartProps {
  performance?: ModelPerformance;
}

const mockPerformance: ModelPerformance = {
  accuracy: 92.3,
  precision: 89.7,
  recall: 94.1,
  f1Score: 91.8
};

const ModelPerformanceChart: React.FC<ModelPerformanceChartProps> = ({
  performance = mockPerformance
}) => {
  // Create data for the donut chart
  const chartData = [
    {
      name: 'Accuracy',
      value: performance.accuracy,
      color: '#4caf50'
    },
    {
      name: 'Remaining',
      value: 100 - performance.accuracy,
      color: '#ffffff20'
    }
  ];

  const getPerformanceColor = (value: number) => {
    if (value >= 90) return '#4caf50'; // Green
    if (value >= 80) return '#ff9800'; // Orange
    if (value >= 70) return '#2196f3'; // Blue
    return '#f44336'; // Red
  };

  const getPerformanceStatus = (accuracy: number) => {
    if (accuracy >= 90) return 'Excellent';
    if (accuracy >= 80) return 'Good';
    if (accuracy >= 70) return 'Fair';
    return 'Needs Improvement';
  };

  return (
    <PerformanceContainer>
      <PerformanceHeader justify="between" align="center">
        <Flex align="center" gap="8px">
          <Brain size={16} />
          <Heading level={5} weight="semibold">Model Performance</Heading>
        </Flex>
        <StatusIndicator>
          <TrendingUp />
          {getPerformanceStatus(performance.accuracy)}
        </StatusIndicator>
      </PerformanceHeader>
      
      <ChartSection>
        <ChartContainer>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={50}
                startAngle={90}
                endAngle={-270}
                dataKey="value"
                strokeWidth={0}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <CenterValue>
            <MainMetric style={{ color: getPerformanceColor(performance.accuracy) }}>
              {performance.accuracy.toFixed(1)}%
            </MainMetric>
            <MetricLabel color="secondary">Accuracy</MetricLabel>
          </CenterValue>
        </ChartContainer>
      </ChartSection>
      
      <MetricsGrid>
        <MetricItem>
          <MetricValue style={{ color: getPerformanceColor(performance.precision) }}>
            {performance.precision.toFixed(1)}%
          </MetricValue>
          <MetricName color="secondary">Precision</MetricName>
        </MetricItem>
        <MetricItem>
          <MetricValue style={{ color: getPerformanceColor(performance.recall) }}>
            {performance.recall.toFixed(1)}%
          </MetricValue>
          <MetricName color="secondary">Recall</MetricName>
        </MetricItem>
        <MetricItem>
          <MetricValue style={{ color: getPerformanceColor(performance.f1Score) }}>
            {performance.f1Score.toFixed(1)}%
          </MetricValue>
          <MetricName color="secondary">F1 Score</MetricName>
        </MetricItem>
        <MetricItem>
          <MetricValue style={{ color: '#2196f3' }}>
            Live
          </MetricValue>
          <MetricName color="secondary">Status</MetricName>
        </MetricItem>
      </MetricsGrid>
    </PerformanceContainer>
  );
};

export default ModelPerformanceChart;