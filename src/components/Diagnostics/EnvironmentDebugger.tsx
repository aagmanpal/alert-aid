import React from 'react';
import styled from 'styled-components';

const DebugContainer = styled.div`
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: rgba(0, 0, 0, 0.9);
  color: #22C55E;
  padding: 16px;
  border-radius: 8px;
  border: 1px solid #22C55E;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  max-width: 400px;
  z-index: 9999;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
`;

const DebugTitle = styled.div`
  font-weight: bold;
  margin-bottom: 8px;
  color: #10B981;
  border-bottom: 1px solid #22C55E;
  padding-bottom: 4px;
`;

const DebugItem = styled.div`
  margin: 4px 0;
  word-break: break-all;
`;

const DebugLabel = styled.span`
  color: #FCD34D;
  font-weight: bold;
`;

const DebugValue = styled.span<{ isSet: boolean }>`
  color: ${props => props.isSet ? '#22C55E' : '#EF4444'};
`;

const CloseButton = styled.button`
  position: absolute;
  top: 8px;
  right: 8px;
  background: transparent;
  border: none;
  color: #9CA3AF;
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
  padding: 4px 8px;
  
  &:hover {
    color: #EF4444;
  }
`;

export const EnvironmentDebugger: React.FC<{ onClose?: () => void }> = ({ onClose }) => {
  const apiUrl = process.env.REACT_APP_API_URL;
  const openWeatherKey = process.env.REACT_APP_OPENWEATHER_API_KEY;
  const nodeEnv = process.env.NODE_ENV;
  
  return (
    <DebugContainer>
      <CloseButton onClick={onClose}>×</CloseButton>
      <DebugTitle>🔧 Environment Debug Info</DebugTitle>
      
      <DebugItem>
        <DebugLabel>NODE_ENV:</DebugLabel>{' '}
        <DebugValue isSet={!!nodeEnv}>{nodeEnv || 'NOT SET'}</DebugValue>
      </DebugItem>
      
      <DebugItem>
        <DebugLabel>REACT_APP_API_URL:</DebugLabel>{' '}
        <DebugValue isSet={!!apiUrl && !apiUrl.includes('localhost')}>
          {apiUrl || 'NOT SET (will use localhost:8000)'}
        </DebugValue>
      </DebugItem>
      
      <DebugItem>
        <DebugLabel>OPENWEATHER_KEY:</DebugLabel>{' '}
        <DebugValue isSet={!!openWeatherKey}>
          {openWeatherKey ? `${openWeatherKey.substring(0, 10)}...` : 'NOT SET'}
        </DebugValue>
      </DebugItem>
      
      <DebugItem style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid #374151' }}>
        <DebugLabel>Status:</DebugLabel>{' '}
        {apiUrl && !apiUrl.includes('localhost') ? (
          <DebugValue isSet={true}>✅ Production Config</DebugValue>
        ) : (
          <DebugValue isSet={false}>⚠️ Using localhost - Env var not set!</DebugValue>
        )}
      </DebugItem>
      
      <div style={{ marginTop: '12px', fontSize: '11px', color: '#6B7280', borderTop: '1px solid #374151', paddingTop: '8px' }}>
        If API_URL shows localhost, the environment variable isn't set correctly in Vercel.
        You must set it, then trigger a NEW BUILD (not just redeploy).
      </div>
    </DebugContainer>
  );
};

export default EnvironmentDebugger;
