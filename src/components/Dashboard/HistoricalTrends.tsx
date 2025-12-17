import React from 'react';
import styled from 'styled-components';
import { Area, AreaChart, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Card, Text, Flex } from '../../styles/components';
import { TrendingUp, TrendingDown } from 'lucide-react';

const TrendsContainer = styled(Card)`
  min-height: 320px;
  display: flex;
  flex-direction: column;
  border: 2px solid ${({ theme }) => theme.colors.primary[100]}; /* Coral border */
  box-shadow: ${({ theme }) => theme.shadows.warmGlow}; /* Warm glow */
`;

const TrendsHeader = styled(Flex)`
  margin-bottom: ${({ theme }) => theme.spacing[4]};
  padding-bottom: ${({ theme }) => theme.spacing[3]};
  border-bottom: 2px solid ${({ theme }) => theme.colors.primary[200]}; /* Coral separator */
`;

const ChartContainer = styled.div`
  flex: 1;
  min-height: 200px;
  position: relative;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: ${({ theme }) => theme.spacing[4]};
  margin-top: ${({ theme }) => theme.spacing[4]};
  padding-top: ${({ theme }) => theme.spacing[4]};
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
`;

const MetricCard = styled.div`
  text-align: center;
`;

const TrendIndicator = styled(Flex)<{ trend: 'up' | 'down' }>`
  color: ${({ trend, theme }) => 
    trend === 'up' ? theme.colors.success[500] : theme.colors.danger[500]
  };
`;

// Mock data for historical trends
const mockHistoricalData = [
  { month: 'Jan', incidents: 45, predictions: 48, accuracy: 94 },
  { month: 'Feb', incidents: 52, predictions: 55, accuracy: 95 },
  { month: 'Mar', incidents: 38, predictions: 40, accuracy: 95 },
  { month: 'Apr', incidents: 67, predictions: 70, accuracy: 96 },
  { month: 'May', incidents: 73, predictions: 75, accuracy: 97 },
  { month: 'Jun', incidents: 58, predictions: 60, accuracy: 97 },
  { month: 'Jul', incidents: 82, predictions: 85, accuracy: 96 },
  { month: 'Aug', incidents: 91, predictions: 95, accuracy: 96 },
  { month: 'Sep', incidents: 76, predictions: 78, accuracy: 97 },
  { month: 'Oct', incidents: 89, predictions: 92, accuracy: 97 },
  { month: 'Nov', incidents: 95, predictions: 98, accuracy: 97 },
  { month: 'Dec', incidents: 103, predictions: 105, accuracy: 98 }
];

interface HistoricalTrendsProps {
  data?: Array<{
    month: string;
    incidents: number;
    predictions: number;
    accuracy: number;
  }>;
}

const HistoricalTrends: React.FC<HistoricalTrendsProps> = ({ 
  data = mockHistoricalData 
}) => {
  const formatTooltip = (value: any, name: string) => {
    if (name === 'incidents') return [`${value}`, 'Actual Incidents'];
    if (name === 'predictions') return [`${value}`, 'Predicted'];
    if (name === 'accuracy') return [`${value}%`, 'Accuracy'];
    return [value, name];
  };

  const currentMonth = data[data.length - 1];
  const previousMonth = data[data.length - 2];
  
  const incidentsTrend = currentMonth.incidents > previousMonth.incidents ? 'up' : 'down';
  const accuracyTrend = currentMonth.accuracy > previousMonth.accuracy ? 'up' : 'down';
  
  const incidentsChange = Math.abs(
    ((currentMonth.incidents - previousMonth.incidents) / previousMonth.incidents) * 100
  );

  return (
    <TrendsContainer>
      <TrendsHeader justify="between" align="center">
        <div>
          <Text size="lg" weight="semibold">Historical Trends</Text>
          <Text size="sm" color="secondary" style={{ marginTop: '2px' }}>
            12-month incident prediction accuracy
          </Text>
        </div>
        <TrendIndicator align="center" gap="4px" trend={accuracyTrend}>
          {accuracyTrend === 'up' ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
          <Text size="sm" weight="medium">{currentMonth.accuracy}%</Text>
        </TrendIndicator>
      </TrendsHeader>

      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
            <defs>
              <linearGradient id="incidentsGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#D72638" stopOpacity={0.4}/> {/* Coral gradient */}
                <stop offset="95%" stopColor="#D72638" stopOpacity={0.08}/>
              </linearGradient>
              <linearGradient id="predictionsGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#FF6F61" stopOpacity={0.4}/> {/* Light coral gradient */}
                <stop offset="95%" stopColor="#FF6F61" stopOpacity={0.08}/>
              </linearGradient>
            </defs>
            
            <XAxis 
              dataKey="month" 
              axisLine={false}
              tickLine={false}
              tick={{ fill: 'rgba(255, 255, 255, 0.7)', fontSize: 12 }}
            />
            <YAxis 
              axisLine={false}
              tickLine={false}
              tick={{ fill: 'rgba(255, 255, 255, 0.7)', fontSize: 12 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                backdropFilter: 'blur(10px)'
              }}
              formatter={formatTooltip}
              labelStyle={{ color: 'rgba(255, 255, 255, 0.9)' }}
            />
            
            <Area
              type="monotone"
              dataKey="incidents"
              stroke="#D72638" // Coral stroke
              strokeWidth={3} // Thicker line
              fill="url(#incidentsGradient)"
            />
            <Area
              type="monotone"
              dataKey="predictions"
              stroke="#FF6F61" // Light coral stroke
              strokeWidth={3} // Thicker line
              fill="url(#predictionsGradient)"
            />
            
            {/* Reference line for current month */}
            <ReferenceLine 
              x={currentMonth.month} 
              stroke="rgba(255, 255, 255, 0.3)" 
              strokeDasharray="5 5" 
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartContainer>

      <MetricsGrid>
        <MetricCard>
          <Text size="sm" color="secondary">Avg Monthly</Text>
          <Text size="lg" weight="semibold" style={{ marginTop: '2px' }}>
            {Math.round(data.reduce((sum, item) => sum + item.incidents, 0) / data.length)}
          </Text>
          <Text size="xs" color="secondary">incidents</Text>
        </MetricCard>
        
        <MetricCard>
          <Text size="sm" color="secondary">Peak Month</Text>
          <Text size="lg" weight="semibold" style={{ marginTop: '2px' }}>
            {Math.max(...data.map(item => item.incidents))}
          </Text>
          <Text size="xs" color="secondary">incidents</Text>
        </MetricCard>
        
        <MetricCard>
          <Text size="sm" color="secondary">Monthly Change</Text>
          <TrendIndicator align="center" gap="4px" trend={incidentsTrend} style={{ justifyContent: 'center', marginTop: '2px' }}>
            {incidentsTrend === 'up' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
            <Text size="lg" weight="semibold">{incidentsChange.toFixed(1)}%</Text>
          </TrendIndicator>
        </MetricCard>
        
        <MetricCard>
          <Text size="sm" color="secondary">Accuracy</Text>
          <Text size="lg" weight="semibold" style={{ marginTop: '2px' }}>
            {(data.reduce((sum, item) => sum + item.accuracy, 0) / data.length).toFixed(1)}%
          </Text>
          <Text size="xs" color="secondary">average</Text>
        </MetricCard>
      </MetricsGrid>
    </TrendsContainer>
  );
};

export default HistoricalTrends;