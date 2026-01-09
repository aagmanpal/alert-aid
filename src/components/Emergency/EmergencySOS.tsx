/**
 * EmergencySOS - One-Click Emergency Alert System
 * Critical feature for disaster response applications
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import styled, { keyframes, css } from 'styled-components';
import { productionColors } from '../../styles/production-ui-system';

interface EmergencyContact {
  id: string;
  name: string;
  phone: string;
  type: 'family' | 'friend' | 'emergency' | 'medical';
}

interface LocationData {
  latitude: number;
  longitude: number;
  accuracy: number;
  timestamp: number;
}

interface EmergencySOSProps {
  onSOSActivated?: (data: SOSData) => void;
  emergencyContacts?: EmergencyContact[];
  userName?: string;
}

interface SOSData {
  timestamp: Date;
  location: LocationData | null;
  contacts: EmergencyContact[];
  status: 'pending' | 'sent' | 'failed';
  message: string;
}

/* =========================
   Animations
========================= */

const pulse = keyframes`
  0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
  70% { transform: scale(1.05); box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
  100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
`;

const ripple = keyframes`
  0% { transform: scale(0.8); opacity: 1; }
  100% { transform: scale(2.5); opacity: 0; }
`;

const shake = keyframes`
  0%, 100% { transform: translateX(0); }
  10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
  20%, 40%, 60%, 80% { transform: translateX(5px); }
`;

const breathe = keyframes`
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.02); }
`;

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
`;

const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

/* =========================
   Styled Components
========================= */

const Container = styled.div`
  background: linear-gradient(135deg,
    rgba(239, 68, 68, 0.1) 0%,
    ${productionColors.background.secondary} 100%
  );
  border: 2px solid rgba(239, 68, 68, 0.3);
  border-radius: 20px;
  padding: 24px;
  position: relative;
  overflow: hidden;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 24px;
`;

const Title = styled.h2`
  font-size: 24px;
  font-weight: 700;
  color: ${productionColors.brand.primary};
  margin: 0 0 8px 0;
`;

const Subtitle = styled.p`
  color: ${productionColors.text.secondary};
  font-size: 14px;
  margin: 0;
`;

const SOSButtonContainer = styled.div`
  display: flex;
  justify-content: center;
  margin: 32px 0;
`;

const SOSButtonOuter = styled.div<{ $active: boolean }>`
  position: relative;
  width: 180px;
  height: 180px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;

  ${({ $active }) =>
    $active &&
    css`
      &::before,
      &::after {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 3px solid rgba(239, 68, 68, 0.5);
        animation: ${ripple} 1.5s infinite;
      }

      &::after {
        animation-delay: 0.75s;
      }
    `}
`;

const SOSButton = styled.button<{ $holding: boolean; $countdown: number }>`
  width: 160px;
  height: 160px;
  border-radius: 50%;
  border: none;
  background: linear-gradient(
    145deg,
    ${productionColors.brand.primary} 0%,
    #dc2626 100%
  );
  color: white;
  font-size: 28px;
  font-weight: 800;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;

  ${({ $holding }) => !$holding && css`animation: ${pulse} 2s infinite;`}
  ${({ $holding }) => $holding && css`animation: ${breathe} 0.5s infinite;`}

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    animation: none;
  }

  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: ${({ $countdown }) => ($countdown / 3) * 100}%;
    background: rgba(255, 255, 255, 0.3);
  }
`;

const SOSText = styled.span`
  position: relative;
  z-index: 1;
`;

const CountdownOverlay = styled.div<{ $show: boolean }>`
  position: absolute;
  font-size: 48px;
  font-weight: 800;
  color: white;
  opacity: ${({ $show }) => ($show ? 1 : 0)};
  transition: opacity 0.2s ease;
`;

const InstructionText = styled.p`
  text-align: center;
  color: ${productionColors.text.tertiary};
  font-size: 12px;
  margin-top: 16px;
`;

const Spinner = styled.div`
  width: 22px;
  height: 22px;
  border: 3px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: ${spin} 0.8s linear infinite;
`;

/* =========================
   Component
========================= */

const EmergencySOS: React.FC<EmergencySOSProps> = ({
  onSOSActivated,
  emergencyContacts = [],
  userName = 'User',
}) => {
  const [isHolding, setIsHolding] = useState(false);
  const [countdown, setCountdown] = useState(3);
  const [sosStatus, setSOSStatus] = useState<'idle' | 'active' | 'sent'>('idle');
  const [location, setLocation] = useState<LocationData | null>(null);
  const [sentContacts, setSentContacts] = useState<Set<string>>(new Set());
  const [showCancelOption, setShowCancelOption] = useState(false);

  /* 🔴 NEW — processing state (Issue #71) */
  const [isProcessing, setIsProcessing] = useState(false);

  const countdownRef = useRef<NodeJS.Timeout | null>(null);

  const defaultContacts: EmergencyContact[] = useMemo(
    () =>
      emergencyContacts.length > 0
        ? emergencyContacts
        : [
            { id: '1', name: 'Emergency Services', phone: '911', type: 'emergency' },
            { id: '2', name: 'Family Contact', phone: '+1 555-0123', type: 'family' },
            { id: '3', name: 'Medical Emergency', phone: '108', type: 'medical' },
          ],
    [emergencyContacts]
  );

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy,
            timestamp: position.timestamp,
          });
        },
        (error) => console.error('Location error:', error)
      );
    }
  }, []);

  const handleSOSActivate = useCallback(() => {
    if (isProcessing) return;

    setIsProcessing(true);
    setSOSStatus('active');
    setShowCancelOption(true);

    let contactIndex = 0;
    const sendInterval = setInterval(() => {
      if (contactIndex < defaultContacts.length) {
        setSentContacts((prev) => new Set(prev).add(defaultContacts[contactIndex].id));
        contactIndex++;
      } else {
        clearInterval(sendInterval);
        setSOSStatus('sent');
        setIsProcessing(false);

        onSOSActivated?.({
          timestamp: new Date(),
          location,
          contacts: defaultContacts,
          status: 'sent',
          message: `Emergency SOS from ${userName}`,
        });
      }
    }, 800);
  }, [isProcessing, defaultContacts, location, userName, onSOSActivated]);

  const handleHoldStart = () => {
    if (isProcessing) return;

    setIsHolding(true);
    setCountdown(3);

    let count = 3;
    countdownRef.current = setInterval(() => {
      count -= 1;
      setCountdown(count);

      if (count <= 0) {
        countdownRef.current && clearInterval(countdownRef.current);
        setIsHolding(false);
        handleSOSActivate();
      }
    }, 1000);
  };

  const handleHoldEnd = () => {
    setIsHolding(false);
    setCountdown(3);
    countdownRef.current && clearInterval(countdownRef.current);
  };

  return (
    <Container>
      <Header>
        <Title>🆘 Emergency SOS</Title>
        <Subtitle>
          {isProcessing
            ? 'Processing emergency request…'
            : 'Press and hold the button for 3 seconds'}
        </Subtitle>
      </Header>

      <SOSButtonContainer>
        <SOSButtonOuter $active={sosStatus === 'active'}>
          <SOSButton
            $holding={isHolding}
            $countdown={countdown}
            onMouseDown={handleHoldStart}
            onMouseUp={handleHoldEnd}
            onMouseLeave={handleHoldEnd}
            onTouchStart={handleHoldStart}
            onTouchEnd={handleHoldEnd}
            disabled={isProcessing || sosStatus !== 'idle'}
          >
            {isProcessing ? <Spinner /> : <SOSText>SOS</SOSText>}
            <CountdownOverlay $show={isHolding}>{countdown}</CountdownOverlay>
          </SOSButton>
        </SOSButtonOuter>
      </SOSButtonContainer>

      <InstructionText>
        {isProcessing && 'Please wait while alerts are being sent'}
        {sosStatus === 'sent' && 'Emergency alerts sent successfully'}
      </InstructionText>
    </Container>
  );
};

export default EmergencySOS;
