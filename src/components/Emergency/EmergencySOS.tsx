/**
 * EmergencySOS - One-Click Emergency Alert System
 * Critical feature for disaster response applications
 * Updated with consistent user-facing error handling (Issue #210)
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import styled, { keyframes, css } from 'styled-components';
import { productionColors } from '../../styles/production-ui-system';

/* ===================== TYPES ===================== */

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

/* ===================== ANIMATIONS ===================== */

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

/* ===================== STYLES ===================== */

const Container = styled.div`
  background: linear-gradient(
    135deg,
    rgba(239, 68, 68, 0.1),
    ${productionColors.background.secondary}
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

const ErrorPanel = styled.div`
  margin-top: 16px;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.4);
  color: #ef4444;
  font-size: 13px;
  animation: ${shake} 0.4s ease;
`;

/* ===== ALL YOUR ORIGINAL STYLED COMPONENTS CONTINUE UNCHANGED ===== */
/* (SOSButton, StatusPanel, Contacts, QuickActions, etc.) */
/* NONE REMOVED — omitted here ONLY FOR COMMENT BREVITY */

/* ===================== COMPONENT ===================== */

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

  // 🔴 NEW: user-facing error state
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const countdownRef = useRef<NodeJS.Timeout | null>(null);

  const defaultContacts = useMemo(
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

  /* ===================== LOCATION ===================== */

  useEffect(() => {
    if (!navigator.geolocation) {
      setErrorMessage('Geolocation is not supported by your browser.');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      position => {
        setLocation({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: position.timestamp,
        });
      },
      () => {
        setErrorMessage(
          'Location access denied. Enable location permissions for accurate emergency alerts.'
        );
      }
    );
  }, []);

  /* ===================== SOS LOGIC ===================== */

  const handleSOSActivate = useCallback(() => {
    setSOSStatus('active');
    setShowCancelOption(true);
    setErrorMessage(null);

    let contactIndex = 0;
    const sendInterval = setInterval(() => {
      if (contactIndex < defaultContacts.length) {
        setSentContacts(prev => new Set(prev).add(defaultContacts[contactIndex].id));
        contactIndex++;
      } else {
        clearInterval(sendInterval);
        setSOSStatus('sent');

        onSOSActivated?.({
          timestamp: new Date(),
          location,
          contacts: defaultContacts,
          status: 'sent',
          message: `Emergency SOS from ${userName}`,
        });
      }
    }, 800);
  }, [defaultContacts, location, onSOSActivated, userName]);

  const handleMouseDown = () => {
    setIsHolding(true);
    let count = 3;
    setCountdown(count);

    countdownRef.current = setInterval(() => {
      count -= 1;
      setCountdown(count);
      if (count <= 0) {
        clearInterval(countdownRef.current!);
        setIsHolding(false);
        handleSOSActivate();
      }
    }, 1000);
  };

  const handleMouseUp = () => {
    setIsHolding(false);
    setCountdown(3);
    if (countdownRef.current) clearInterval(countdownRef.current);
  };

  /* ===================== RENDER ===================== */

  return (
    <Container>
      <Header>
        <Title>🆘 Emergency SOS</Title>
        <Subtitle>Press and hold the button for 3 seconds</Subtitle>
      </Header>

      {/* ===== YOUR EXISTING SOS BUTTON UI (UNCHANGED) ===== */}

      {errorMessage && <ErrorPanel>{errorMessage}</ErrorPanel>}

      {/* ===== ALL REMAINING UI: STATUS, CONTACTS, QUICK ACTIONS (UNCHANGED) ===== */}
    </Container>
  );
};

export default EmergencySOS;
