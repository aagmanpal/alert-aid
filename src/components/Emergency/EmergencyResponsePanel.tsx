import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { Card, Button } from '../../styles/components';

// Types for emergency system
interface EmergencyContact {
  id: string;
  name: string;
  phone: string;
  relationship: 'family' | 'friend' | 'medical' | 'emergency-service';
  isPrimary: boolean;
}

interface MedicalInfo {
  bloodType: string;
  allergies: string[];
  medications: string[];
  conditions: string[];
  emergencyNotes: string;
  emergencyContact: string;
  lastUpdated: number;
}

interface SOSState {
  isActive: boolean;
  startTime: number | null;
  countdown: number;
  locationSent: boolean;
  contactsNotified: string[];
}

// Styled components
const EmergencyContainer = styled(Card)`
  background: ${({ theme }) => theme.colors.surface.elevated};
  border: 2px solid ${({ theme }) => theme.colors.danger[600]};
  padding: ${({ theme }) => theme.spacing.lg};
`;

const SOSButton = styled.button<{ isActive: boolean }>`
  width: 120px;
  height: 120px;
  border-radius: 50%;
  border: none;
  font-size: 1.2rem;
  font-weight: bold;
  cursor: pointer;
  position: relative;
  margin: ${({ theme }) => theme.spacing.lg} auto;
  display: block;
  transition: all 0.3s ease;
  
  background: ${({ theme, isActive }) => 
    isActive ? theme.colors.danger[700] : theme.colors.danger[600]
  };
  color: ${({ theme }) => theme.colors.text.inverse};
  
  &:hover {
    transform: scale(1.05);
    background: ${({ theme }) => theme.colors.danger[700]};
  }
  
  &:active {
    transform: scale(0.95);
  }
  
  ${({ isActive }) => isActive && `
    animation: pulse 1s infinite;
    
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(215, 38, 56, 0.7); }
      70% { box-shadow: 0 0 0 20px rgba(215, 38, 56, 0); }
      100% { box-shadow: 0 0 0 0 rgba(215, 38, 56, 0); }
    }
  `}
`;

const CountdownOverlay = styled.div<{ countdown: number }>`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 2rem;
  font-weight: bold;
  color: ${({ theme }) => theme.colors.text.inverse};
  
  ${({ countdown }) => countdown <= 3 && `
    animation: urgentFlash 0.5s infinite;
    
    @keyframes urgentFlash {
      0%, 50% { opacity: 1; }
      51%, 100% { opacity: 0.3; }
    }
  `}
`;

const EmergencySection = styled.div`
  margin: ${({ theme }) => theme.spacing.lg} 0;
  
  h3 {
    color: ${({ theme }) => theme.colors.text.primary};
    margin-bottom: ${({ theme }) => theme.spacing.md};
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: ${({ theme }) => theme.spacing.xs};
  }
`;

const ContactsList = styled.div`
  display: grid;
  gap: ${({ theme }) => theme.spacing.sm};
`;

const ContactItem = styled.div`
  display: flex;
  align-items: center;
  justify-content: between;
  padding: ${({ theme }) => theme.spacing.md};
  background: ${({ theme }) => theme.colors.surface.default};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 1px solid ${({ theme }) => theme.colors.surface.border};
`;

const ContactInfo = styled.div`
  flex: 1;
  
  .name {
    color: ${({ theme }) => theme.colors.text.primary};
    font-weight: 500;
    margin-bottom: 2px;
  }
  
  .phone {
    color: ${({ theme }) => theme.colors.text.secondary};
    font-size: 0.9rem;
  }
  
  .relationship {
    color: ${({ theme }) => theme.colors.text.caption};
    font-size: 0.8rem;
    text-transform: capitalize;
  }
`;

const QuickCallButton = styled(Button)`
  margin-left: ${({ theme }) => theme.spacing.sm};
  padding: ${({ theme }) => theme.spacing.sm};
  min-width: auto;
`;

const StatusMessage = styled.div<{ type: 'info' | 'success' | 'warning' | 'error' }>`
  padding: ${({ theme }) => theme.spacing.md};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  margin: ${({ theme }) => theme.spacing.md} 0;
  
  background: ${({ theme, type }) => 
    type === 'success' ? theme.colors.success[100] :
    type === 'warning' ? theme.colors.warning[100] :
    type === 'error' ? theme.colors.danger[100] :
    theme.colors.surface.panel
  };
  
  color: ${({ theme, type }) => 
    type === 'success' ? theme.colors.success[700] :
    type === 'warning' ? theme.colors.warning[700] :
    type === 'error' ? theme.colors.danger[700] :
    theme.colors.text.secondary
  };
  
  border: 1px solid ${({ theme, type }) => 
    type === 'success' ? theme.colors.success[300] :
    type === 'warning' ? theme.colors.warning[300] :
    type === 'error' ? theme.colors.danger[300] :
    theme.colors.surface.border
  };
`;

// Emergency Response Panel Component
const EmergencyResponsePanel: React.FC = () => {
  const [sosState, setSosState] = useState<SOSState>({
    isActive: false,
    startTime: null,
    countdown: 0,
    locationSent: false,
    contactsNotified: []
  });

  const [emergencyContacts, setEmergencyContacts] = useState<EmergencyContact[]>([
    {
      id: '1',
      name: 'Emergency Services',
      phone: '112',
      relationship: 'emergency-service',
      isPrimary: true
    },
    {
      id: '2', 
      name: 'Police',
      phone: '100',
      relationship: 'emergency-service',
      isPrimary: false
    },
    {
      id: '3',
      name: 'Fire Department',
      phone: '101',
      relationship: 'emergency-service',
      isPrimary: false
    },
    {
      id: '4',
      name: 'Ambulance',
      phone: '102',
      relationship: 'emergency-service',
      isPrimary: false
    },
    {
      id: '5',
      name: 'Disaster Management',
      phone: '108',
      relationship: 'emergency-service',
      isPrimary: false
    }
  ]);

  // Medical info removed as requested - keeping minimal structure for compatibility
  const [medicalInfo] = useState<MedicalInfo>({
    bloodType: '',
    allergies: [],
    medications: [],
    conditions: [],
    emergencyNotes: '',
    emergencyContact: '',
    lastUpdated: Date.now()
  });

  const [status, setStatus] = useState<{ message: string; type: 'info' | 'success' | 'warning' | 'error' } | null>(null);

  // Execute SOS actions - defined before useEffect to avoid hoisting issues
  const executeSOS = async () => {
    try {
      setStatus({ message: 'Executing emergency protocol...', type: 'warning' });
      
      // Get current location with high accuracy
      const location = await getCurrentLocation();
      
      // Send location to emergency contacts via multiple channels
      await notifyEmergencyContacts(location);
      
      // Save emergency event to localStorage for recovery
      const emergencyEvent = {
        timestamp: Date.now(),
        location,
        medicalInfo: medicalInfo,
        contactsNotified: emergencyContacts.map(c => c.id),
        status: 'active'
      };
      localStorage.setItem('activeEmergency', JSON.stringify(emergencyEvent));
      
      // Call primary emergency number (112)
      const primaryContact = emergencyContacts.find(c => c.isPrimary);
      if (primaryContact) {
        setTimeout(() => {
          window.open(`tel:${primaryContact.phone}`);
        }, 1000); // Small delay to let location send first
      }
      
      // Update state with success
      setSosState(prev => ({ 
        ...prev, 
        locationSent: true, 
        contactsNotified: emergencyContacts.map(c => c.id) 
      }));
      
      // Vibrate phone if supported
      if ('vibrate' in navigator) {
        navigator.vibrate([200, 100, 200, 100, 200]);
      }
      
      setStatus({ 
        message: '✅ SOS SENT! Emergency services contacted and location shared.', 
        type: 'success' 
      });
      
      // Keep success message visible longer
      setTimeout(() => {
        if (sosState.isActive) {
          setStatus({ 
            message: 'Emergency active. Help is on the way. Stay calm and safe.', 
            type: 'info' 
          });
        }
      }, 5000);
      
    } catch (error) {
      console.error('SOS execution failed:', error);
      
      // Even if location fails, still make the call
      const primaryContact = emergencyContacts.find(c => c.isPrimary);
      if (primaryContact) {
        window.open(`tel:${primaryContact.phone}`);
      }
      
      setStatus({ 
        message: '⚠️ SOS partially sent. Location failed but calling emergency services. Please provide location verbally.', 
        type: 'error' 
      });
    }
  };

  // SOS countdown timer
  useEffect(() => {
    let timer: NodeJS.Timeout;
    
    if (sosState.isActive && sosState.countdown > 0) {
      timer = setTimeout(() => {
        setSosState(prev => ({ ...prev, countdown: prev.countdown - 1 }));
      }, 1000);
    } else if (sosState.isActive && sosState.countdown === 0) {
      // Execute SOS actions
      executeSOS();
    }
    
    return () => clearTimeout(timer);
  }, [sosState.isActive, sosState.countdown, executeSOS]);

  // Start SOS sequence
  const startSOS = () => {
    setStatus({ message: 'SOS activated! Canceling in 10 seconds...', type: 'warning' });
    setSosState({
      isActive: true,
      startTime: Date.now(),
      countdown: 10,
      locationSent: false,
      contactsNotified: []
    });
  };

  // Cancel SOS
  const cancelSOS = () => {
    setSosState({
      isActive: false,
      startTime: null,
      countdown: 0,
      locationSent: false,
      contactsNotified: []
    });
    setStatus({ message: 'SOS canceled', type: 'info' });
    setTimeout(() => setStatus(null), 3000);
  };

  // Get current location for SOS with enhanced accuracy
  const getCurrentLocation = (): Promise<{ lat: number; lon: number; accuracy: number; address?: string }> => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation not supported by this device'));
        return;
      }
      
      // First try high accuracy GPS
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const location = {
            lat: position.coords.latitude,
            lon: position.coords.longitude,
            accuracy: position.coords.accuracy
          };
          
          // Try to get human-readable address
          try {
            const address = await reverseGeocode(location.lat, location.lon);
            resolve({ ...location, address });
          } catch {
            resolve(location); // Return without address if geocoding fails
          }
        },
        (error) => {
          console.error('High accuracy GPS failed:', error);
          
          // Fallback to lower accuracy
          navigator.geolocation.getCurrentPosition(
            (position) => {
              resolve({
                lat: position.coords.latitude,
                lon: position.coords.longitude,
                accuracy: position.coords.accuracy
              });
            },
            (fallbackError) => {
              console.error('All location methods failed:', fallbackError);
              reject(new Error(`Location unavailable: ${fallbackError.message}`));
            },
            { enableHighAccuracy: false, timeout: 10000, maximumAge: 60000 }
          );
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000 }
      );
    });
  };

  // Reverse geocode coordinates to address
  const reverseGeocode = async (lat: number, lon: number): Promise<string> => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&addressdetails=1`
      );
      const data = await response.json();
      return data.display_name || `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    } catch {
      return `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    }
  };

  // Notify emergency contacts via multiple methods
  const notifyEmergencyContacts = async (location: { lat: number; lon: number; accuracy?: number; address?: string }) => {
    const timestamp = new Date().toLocaleString();
    const accuracyText = location.accuracy ? ` (±${Math.round(location.accuracy)}m)` : '';
    const locationText = location.address || `${location.lat.toFixed(6)}, ${location.lon.toFixed(6)}`;
    
    const emergencyMessage = `🚨 EMERGENCY ALERT from Alert Aid App
    
Time: ${timestamp}
I need immediate assistance at:
📍 ${locationText}${accuracyText}
🗺️ Maps: https://maps.google.com/maps?q=${location.lat},${location.lon}

This is an automated emergency alert. Please respond immediately.`;

    try {
      // Method 1: Web Share API (if available)
      if (navigator.share) {
        try {
          await navigator.share({
            title: '🚨 EMERGENCY ALERT',
            text: emergencyMessage,
            url: `https://maps.google.com/maps?q=${location.lat},${location.lon}`
          });
          console.log('Emergency alert shared via Web Share API');
        } catch (shareError) {
          console.log('Web Share failed, trying other methods');
        }
      }

      // Method 2: Copy to clipboard for manual sharing
      if (navigator.clipboard) {
        try {
          await navigator.clipboard.writeText(emergencyMessage);
          setStatus({ 
            message: '📋 Emergency details copied to clipboard. Share with contacts!', 
            type: 'success' 
          });
        } catch (clipboardError) {
          console.error('Clipboard write failed:', clipboardError);
        }
      }

      // Method 3: SMS links for each contact (opens default SMS app)
      const personalContacts = emergencyContacts.filter(c => c.relationship !== 'emergency-service');
      personalContacts.forEach((contact, index) => {
        setTimeout(() => {
          const smsBody = encodeURIComponent(emergencyMessage);
          window.open(`sms:${contact.phone}?body=${smsBody}`);
        }, index * 1000); // Stagger SMS opens to avoid blocking
      });

      // Method 4: Email as fallback (if configured)
      const emailSubject = encodeURIComponent('🚨 EMERGENCY ALERT - Immediate Assistance Needed');
      const emailBody = encodeURIComponent(emergencyMessage);
      setTimeout(() => {
        window.open(`mailto:?subject=${emailSubject}&body=${emailBody}`);
      }, personalContacts.length * 1000);

      // Log for emergency recovery
      const notificationLog = {
        timestamp: Date.now(),
        location,
        message: emergencyMessage,
        contactsAttempted: emergencyContacts.map(c => ({ name: c.name, phone: c.phone })),
        methods: ['clipboard', 'sms', 'email']
      };
      localStorage.setItem('lastEmergencyNotification', JSON.stringify(notificationLog));

      return true;
    } catch (error) {
      console.error('Emergency notification failed:', error);
      
      // Fallback: show alert dialog with emergency info
      alert(`EMERGENCY ALERT!\n\n${emergencyMessage}\n\nPlease manually contact emergency services and share this information.`);
      
      throw error;
    }
  };

  // Quick call function with enhanced features
  const makeCall = (phone: string, contactName?: string) => {
    // Log the call for emergency records
    const callLog = {
      timestamp: Date.now(),
      contact: contactName || phone,
      phone: phone,
      context: sosState.isActive ? 'emergency' : 'normal'
    };
    
    const existingLogs = JSON.parse(localStorage.getItem('emergencyCallLogs') || '[]');
    existingLogs.push(callLog);
    localStorage.setItem('emergencyCallLogs', JSON.stringify(existingLogs.slice(-50))); // Keep last 50 calls
    
    // Show feedback
    setStatus({ 
      message: `Calling ${contactName || phone}...`, 
      type: 'info' 
    });
    setTimeout(() => setStatus(null), 2000);
    
    // Make the call
    window.open(`tel:${phone}`);
  };

  // Load data from localStorage
  useEffect(() => {
    const savedContacts = localStorage.getItem('emergencyContacts');
    
    if (savedContacts) {
      setEmergencyContacts(JSON.parse(savedContacts));
    }
  }, []);

  // Save data to localStorage when changed
  useEffect(() => {
    localStorage.setItem('emergencyContacts', JSON.stringify(emergencyContacts));
  }, [emergencyContacts]);

  return (
    <EmergencyContainer>
      <EmergencySection>
        <h2 style={{ textAlign: 'center', marginBottom: '24px', color: '#F7F7FA' }}>
          🚨 Emergency Response
        </h2>
        
        {/* Enhanced SOS Button */}
        <SOSButton 
          isActive={sosState.isActive}
          onClick={sosState.isActive ? cancelSOS : startSOS}
          onDoubleClick={() => {
            if (!sosState.isActive) {
              // Double-click for immediate SOS (bypassing countdown)
              setStatus({ message: 'Double-click detected! Immediate SOS activated!', type: 'error' });
              setSosState({
                isActive: true,
                startTime: Date.now(),
                countdown: 1, // Nearly immediate
                locationSent: false,
                contactsNotified: []
              });
            }
          }}
        >
          {sosState.isActive ? (
            <CountdownOverlay countdown={sosState.countdown}>
              {sosState.countdown > 3 ? sosState.countdown : '🚨'}
            </CountdownOverlay>
          ) : (
            <>
              🆘<br />SOS
            </>
          )}
        </SOSButton>
        
        <p style={{ textAlign: 'center', color: '#D0CFD5', fontSize: '0.9rem', marginBottom: '24px' }}>
          {sosState.isActive 
            ? `⏱️ Emergency alert in ${sosState.countdown}s - Click to cancel`
            : '🆘 Click to start SOS countdown • Double-click for immediate SOS'
          }
        </p>
        
        {status && (
          <StatusMessage type={status.type}>
            {status.message}
          </StatusMessage>
        )}
      </EmergencySection>

      {/* Emergency Contacts */}
      <EmergencySection>
        <h3>📞 Emergency Contacts</h3>
        <ContactsList>
          {emergencyContacts.map(contact => (
            <ContactItem key={contact.id}>
              <ContactInfo>
                <div className="name">{contact.name}</div>
                <div className="phone">{contact.phone}</div>
                <div className="relationship">{contact.relationship.replace('-', ' ')}</div>
              </ContactInfo>
              <QuickCallButton
                variant="primary"
                onClick={() => makeCall(contact.phone, contact.name)}
              >
                📞 Call
              </QuickCallButton>
            </ContactItem>
          ))}
        </ContactsList>
      </EmergencySection>

      {/* Medical Information section removed as requested */}
    </EmergencyContainer>
  );
};

export default EmergencyResponsePanel;