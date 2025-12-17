import React from 'react';
import { Shield, TrendingUp, TrendingDown } from 'lucide-react';
import { Card, CardHeader, CardContent, MetricValue, ProgressRing, Badge, IconWrapper } from './CardStyles';

interface RiskScoreCardProps {
  prediction: any;
}

const RiskScoreCard: React.FC<RiskScoreCardProps> = ({ prediction }) => {
  const riskScore = prediction?.risk_score || 0;
  const confidence = prediction?.confidence || 0;
  const overallRisk = prediction?.overall_risk || 'low';
  
  const getRiskColor = (score: number) => {
    if (score >= 8) return '#f56565';
    if (score >= 6) return '#fbd38d';
    if (score >= 4) return '#fbd38d';
    return '#68d391';
  };

  const getRiskVariant = (risk: string) => {
    switch (risk) {
      case 'critical': return 'danger';
      case 'high': return 'warning';
      case 'moderate': return 'warning';
      default: return 'success';
    }
  };

  const getTrendIcon = () => {
    // Mock trend data - in real app, this would come from API
    const isIncreasing = Math.random() > 0.5;
    return isIncreasing ? <TrendingUp size={16} /> : <TrendingDown size={16} />;
  };

  return (
    <Card variant={getRiskVariant(overallRisk)} size="large" animate={riskScore >= 7}>
      <CardHeader>
        <h3>
          <IconWrapper variant={getRiskVariant(overallRisk)}>
            <Shield size={20} />
          </IconWrapper>
          Overall Risk Score
        </h3>
        <Badge variant={getRiskVariant(overallRisk)}>
          {getTrendIcon()}
          {overallRisk.toUpperCase()}
        </Badge>
      </CardHeader>
      
      <CardContent>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '2rem' }}>
          <div style={{ position: 'relative' }}>
            <ProgressRing 
              percentage={(riskScore / 10) * 100} 
              size={120} 
              strokeWidth={12}
              color={getRiskColor(riskScore)}
            >
              <svg width={120} height={120}>
                <circle
                  className="progress-ring-background"
                  cx={60}
                  cy={60}
                  r={48}
                />
                <circle
                  className="progress-ring-progress"
                  cx={60}
                  cy={60}
                  r={48}
                />
              </svg>
            </ProgressRing>
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center'
            }}>
              <MetricValue size="large" style={{ color: getRiskColor(riskScore) }}>
                {riskScore.toFixed(1)}
              </MetricValue>
              <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>/ 10</div>
            </div>
          </div>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'center' }}>
          <div>
            <MetricValue size="small">{Math.round(confidence * 100)}%</MetricValue>
            <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>Confidence</div>
          </div>
          <div>
            <MetricValue size="small">{prediction?.model_version || 'v2.0'}</MetricValue>
            <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>Model Version</div>
          </div>
        </div>
        
        {prediction?.prediction_timestamp && (
          <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: '#718096', textAlign: 'center' }}>
            Last updated: {new Date(prediction.prediction_timestamp).toLocaleTimeString()}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RiskScoreCard;