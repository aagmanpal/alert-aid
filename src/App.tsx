import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import styled from 'styled-components';
import { theme } from './styles/theme';
import { GlobalStyles } from './styles/GlobalStyles';
import { NotificationProvider } from './contexts/NotificationContext';
import { LocationProvider, useLocation } from './contexts/LocationContext';
import { GeolocationProvider } from './components/Location/GeolocationManager';
import { ToastProvider } from './components/Notifications/ToastSystem';
import Starfield from './components/Starfield/Starfield';
import { NavigationBar } from './components/navigation/NavigationBar';
import Dashboard from './components/Dashboard/Dashboard';
import HomePage from './pages/HomePage';
import PredictionsPage from './pages/PredictionsPage';
import AlertsPage from './pages/AlertsPage';
import EvacuationPage from './pages/EvacuationPage';
import VerificationDashboard from './components/Verification/VerificationDashboard';
import EnhancedLocationPermissionModal from './components/Location/EnhancedLocationPermissionModal';
import logger from './utils/logger';
import { productionColors } from './styles/production-ui-system';
// Import location override utility
import './utils/locationOverride';

// Skip to content link for accessibility
const SkipToContent = styled.a`
  position: absolute;
  top: -40px;
  left: 0;
  background: ${productionColors.brand.primary};
  color: ${productionColors.text.primary};
  padding: 8px 16px;
  text-decoration: none;
  z-index: 10000;
  border-radius: 0 0 4px 0;
  font-weight: 600;

  &:focus {
    top: 0;
    outline: 2px solid ${productionColors.border.accent};
    outline-offset: 2px;
  }
`;

// Utility function to clean up old/invalid location caches (runs only when needed)
const cleanupInvalidCaches = () => {
  try {
    // Only remove clearly invalid or corrupted cache entries
    const keysToCheck = ['enhanced-location-cache', 'location-override'];
    
    keysToCheck.forEach(key => {
      const value = localStorage.getItem(key);
      if (value) {
        try {
          JSON.parse(value);
        } catch (e) {
          // Remove corrupted cache
          logger.warn(`Removing corrupted cache: ${key}`);
          localStorage.removeItem(key);
        }
      }
    });
  } catch (error) {
    logger.warn('Failed to cleanup caches:', error);
  }
};

// Main App Content Component
const AppContent: React.FC = () => {
  const { showLocationModal, setLocation } = useLocation();
  const [currentPage, setCurrentPage] = React.useState('home');
  
  return (
    <>
      <SkipToContent href="#main-content">
        Skip to main content
      </SkipToContent>
      <div className="App" id="main-content">
        <NavigationBar 
          currentPage={currentPage}
          onNavigate={(page) => {
            setCurrentPage(page);
            window.location.href = `/${page === 'home' ? '' : page}`;
          }}
        />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/predictions" element={<PredictionsPage />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/evacuation" element={<EvacuationPage />} />
          <Route path="/verify" element={<VerificationDashboard />} />
        </Routes>
      </div>
      
      {/* Location Permission Modal - Blocks dashboard until location is set */}
      <EnhancedLocationPermissionModal
        isOpen={showLocationModal}
        onLocationGranted={setLocation}
        onClose={() => {}} // No close - modal is required
      />
    </>
  );
};

function App() {
  // Clean up invalid caches only on mount (not clearing valid data)
  useEffect(() => {
    cleanupInvalidCaches();
    logger.log('✅ App mounted - location will be managed by LocationProvider');
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyles theme={theme} />
      <ToastProvider>
        <NotificationProvider>
          <LocationProvider>
            <GeolocationProvider>
              <Router>
                {/* Global starfield sits behind everything */}
                <Starfield />
                <AppContent />
              </Router>
            </GeolocationProvider>
          </LocationProvider>
        </NotificationProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}

export default App;
