import React from 'react';
import { Brain, Activity, CheckCircle } from 'lucide-react';
import { Card, CardHeader, CardContent, MetricValue, ProgressRing, Badge, IconWrapper, FlexRow } from './CardStyles';

interface ModelAccuracyCardProps {
  prediction: any;
}

const ModelAccuracyCard: React.FC<ModelAccuracyCardProps> = ({ prediction }) => {
  // Mock model performance metrics - in real app, these would come from ML monitoring
  const modelAccuracy = 94.2;
  const predictionLatency = 45; // ms
  const dataQuality = 98.7;
  const modelStatus = 'healthy';
  
  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'danger';
      default: return 'info';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle size={16} />;
      case 'warning': return <Activity size={16} />;
      default: return <Brain size={16} />;
    }
  };

  return (
    <Card variant={getStatusVariant(modelStatus)} size="medium">
      <CardHeader>
        <h3>
          <IconWrapper variant={getStatusVariant(modelStatus)}>
            <Brain size={20} />
          </IconWrapper>
          ML Model Performance
        </h3>
        <Badge variant={getStatusVariant(modelStatus)}>
          {getStatusIcon(modelStatus)}
          {modelStatus.toUpperCase()}
        </Badge>
      </CardHeader>
      
      <CardContent>
        <div style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
            <div style={{ position: 'relative' }}>
              <ProgressRing 
                percentage={modelAccuracy} 
                size={100} 
                strokeWidth={8}
                color="#68d391"
              >
                <svg width={100} height={100}>
                  <circle
                    className="progress-ring-background"
                    cx={50}
                    cy={50}
                    r={42}
                  />
                  <circle
                    className="progress-ring-progress"
                    cx={50}
                    cy={50}
                    r={42}
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
                <MetricValue size="medium" style={{ color: '#68d391', fontSize: '1.8rem' }}>
                  {modelAccuracy}%
                </MetricValue>
                <div style={{ fontSize: '0.7rem', color: '#a0aec0' }}>Accuracy</div>
              </div>
            </div>
          </div>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#4fd1c7' }}>{predictionLatency}ms</MetricValue>
            <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>Latency</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <MetricValue size="small" style={{ color: '#fbd38d' }}>{dataQuality}%</MetricValue>
            <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>Data Quality</div>
          </div>
        </div>
        
        <FlexRow gap="0.5rem" justify="space-between" style={{ marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.9rem', color: '#cbd5e0' }}>Model Version:</span>
          <span style={{ fontSize: '0.9rem', color: '#4fd1c7', fontWeight: '600' }}>
            {prediction?.model_version || 'RandomForest_v2.0'}
          </span>
        </FlexRow>
        
        <FlexRow gap="0.5rem" justify="space-between" style={{ marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.9rem', color: '#cbd5e0' }}>Confidence:</span>
          <span style={{ fontSize: '0.9rem', color: '#68d391', fontWeight: '600' }}>
            {Math.round((prediction?.confidence || 0) * 100)}%
          </span>
        </FlexRow>
        
        <FlexRow gap="0.5rem" justify="space-between">
          <span style={{ fontSize: '0.9rem', color: '#cbd5e0' }}>Data Sources:</span>
          <span style={{ fontSize: '0.9rem', color: '#fbd38d', fontWeight: '600' }}>
            5 Active
          </span>
        </FlexRow>
        
        <div style={{ 
          marginTop: '1.5rem', 
          padding: '0.75rem', 
          background: 'rgba(72, 187, 120, 0.1)', 
          borderRadius: '8px',
          border: '1px solid rgba(72, 187, 120, 0.2)'
        }}>
          <p style={{ margin: '0', fontSize: '0.8rem', color: '#68d391' }}>
            ✅ Model is performing optimally with real-time data integration
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelAccuracyCard;