import React, { useState } from 'react';
import styled from 'styled-components';

interface ManualLocationSearchProps {
  onLocationSelect: (location: { latitude: number; longitude: number; city?: string; country?: string }) => void;
  onClose: () => void;
}

const ManualLocationSearch: React.FC<ManualLocationSearchProps> = ({ onLocationSelect, onClose }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [coordinatesMode, setCoordinatesMode] = useState(false);
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');

  const searchByCity = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);

    try {
      // Use OpenWeatherMap Geocoding API
      const apiKey = '1801423b3942e324ab80f5b47afe0859';
      const response = await fetch(
        `https://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(searchQuery)}&limit=5&appid=${apiKey}`
      );

      if (!response.ok) {
        throw new Error('Failed to search location');
      }

      const data = await response.json();
      
      if (data.length === 0) {
        setError('No locations found. Try a different search term.');
      } else {
        setSearchResults(data);
      }
    } catch (err) {
      setError('Error searching location. Please try again.');
      console.error('Location search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCitySearch = (e: React.FormEvent) => {
    e.preventDefault();
    searchByCity();
  };

  const handleCoordinatesSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const lat = parseFloat(latitude);
    const lon = parseFloat(longitude);

    if (isNaN(lat) || isNaN(lon)) {
      setError('Please enter valid coordinates');
      return;
    }

    if (lat < -90 || lat > 90) {
      setError('Latitude must be between -90 and 90');
      return;
    }

    if (lon < -180 || lon > 180) {
      setError('Longitude must be between -180 and 180');
      return;
    }

    onLocationSelect({
      latitude: lat,
      longitude: lon,
      city: `Custom Location`,
      country: ''
    });
  };

  const handleResultClick = (result: any) => {
    onLocationSelect({
      latitude: result.lat,
      longitude: result.lon,
      city: result.name,
      country: result.country
    });
  };

  return (
    <Overlay onClick={onClose}>
      <Modal onClick={(e) => e.stopPropagation()}>
        <Header>
          <Title>📍 Manual Location Search</Title>
          <CloseButton onClick={onClose}>&times;</CloseButton>
        </Header>

        <ModeToggle>
          <ModeButton 
            active={!coordinatesMode} 
            onClick={() => setCoordinatesMode(false)}
          >
            🏙️ Search by City
          </ModeButton>
          <ModeButton 
            active={coordinatesMode} 
            onClick={() => setCoordinatesMode(true)}
          >
            🗺️ Enter Coordinates
          </ModeButton>
        </ModeToggle>

        {!coordinatesMode ? (
          <>
            <SearchForm onSubmit={handleCitySearch}>
              <SearchInput
                type="text"
                placeholder="Enter city name (e.g., Mumbai, Delhi, Jaipur)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <SearchButton type="submit" disabled={loading}>
                {loading ? '🔄 Searching...' : '🔍 Search'}
              </SearchButton>
            </SearchForm>

            {error && <ErrorMessage>{error}</ErrorMessage>}

            {searchResults.length > 0 && (
              <ResultsContainer>
                <ResultsTitle>Search Results:</ResultsTitle>
                {searchResults.map((result, index) => (
                  <ResultItem key={index} onClick={() => handleResultClick(result)}>
                    <LocationName>{result.name}</LocationName>
                    <LocationDetails>
                      {result.state && `${result.state}, `}
                      {result.country}
                    </LocationDetails>
                    <Coordinates>
                      📍 {result.lat.toFixed(4)}°, {result.lon.toFixed(4)}°
                    </Coordinates>
                  </ResultItem>
                ))}
              </ResultsContainer>
            )}
          </>
        ) : (
          <CoordinatesForm onSubmit={handleCoordinatesSubmit}>
            <CoordinatesInfo>
              Enter latitude and longitude coordinates directly
            </CoordinatesInfo>
            
            <CoordinateInputGroup>
              <CoordinateLabel>Latitude (-90 to 90):</CoordinateLabel>
              <CoordinateInput
                type="number"
                step="0.0001"
                placeholder="e.g., 26.9124"
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                required
              />
            </CoordinateInputGroup>

            <CoordinateInputGroup>
              <CoordinateLabel>Longitude (-180 to 180):</CoordinateLabel>
              <CoordinateInput
                type="number"
                step="0.0001"
                placeholder="e.g., 75.7873"
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                required
              />
            </CoordinateInputGroup>

            {error && <ErrorMessage>{error}</ErrorMessage>}

            <SearchButton type="submit">
              ✅ Set Location
            </SearchButton>
          </CoordinatesForm>
        )}

        <HelpText>
          💡 Tip: Use city search for easy location finding, or enter precise coordinates for exact positioning.
        </HelpText>
      </Modal>
    </Overlay>
  );
};

// Styled Components
const Overlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
  backdrop-filter: blur(4px);
  animation: fadeIn 0.2s ease-out;

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
`;

const Modal = styled.div`
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border-radius: 16px;
  padding: 32px;
  max-width: 600px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
  animation: slideUp 0.3s ease-out;

  @keyframes slideUp {
    from {
      transform: translateY(30px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  &::-webkit-scrollbar {
    width: 8px;
  }

  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;

    &:hover {
      background: rgba(255, 255, 255, 0.3);
    }
  }
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
`;

const Title = styled.h2`
  color: #fff;
  font-size: 24px;
  font-weight: 700;
  margin: 0;
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  color: #fff;
  font-size: 36px;
  cursor: pointer;
  padding: 0;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  transition: all 0.2s;

  &:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: scale(1.1);
  }
`;

const ModeToggle = styled.div`
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
`;

const ModeButton = styled.button<{ active: boolean }>`
  flex: 1;
  padding: 12px;
  background: ${props => props.active ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'rgba(255, 255, 255, 0.1)'};
  border: ${props => props.active ? '2px solid #667eea' : '2px solid rgba(255, 255, 255, 0.2)'};
  border-radius: 10px;
  color: #fff;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;

  &:hover {
    background: ${props => props.active ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'rgba(255, 255, 255, 0.15)'};
    transform: translateY(-2px);
  }
`;

const SearchForm = styled.form`
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
`;

const SearchInput = styled.input`
  flex: 1;
  padding: 14px 18px;
  background: rgba(255, 255, 255, 0.1);
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 10px;
  color: #fff;
  font-size: 15px;
  transition: all 0.3s;

  &::placeholder {
    color: rgba(255, 255, 255, 0.5);
  }

  &:focus {
    outline: none;
    border-color: #667eea;
    background: rgba(255, 255, 255, 0.15);
  }
`;

const SearchButton = styled.button`
  padding: 14px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 10px;
  color: #fff;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  white-space: nowrap;

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ErrorMessage = styled.div`
  padding: 12px 16px;
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid #ef4444;
  border-radius: 8px;
  color: #fca5a5;
  font-size: 14px;
  margin-bottom: 16px;
`;

const ResultsContainer = styled.div`
  margin-top: 24px;
`;

const ResultsTitle = styled.h3`
  color: #fff;
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 12px;
`;

const ResultItem = styled.div`
  padding: 16px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  margin-bottom: 12px;
  cursor: pointer;
  transition: all 0.3s;

  &:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: #667eea;
    transform: translateX(4px);
  }
`;

const LocationName = styled.div`
  color: #fff;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 4px;
`;

const LocationDetails = styled.div`
  color: rgba(255, 255, 255, 0.7);
  font-size: 14px;
  margin-bottom: 6px;
`;

const Coordinates = styled.div`
  color: rgba(255, 255, 255, 0.5);
  font-size: 13px;
  font-family: 'Courier New', monospace;
`;

const CoordinatesForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const CoordinatesInfo = styled.div`
  color: rgba(255, 255, 255, 0.7);
  font-size: 14px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  text-align: center;
`;

const CoordinateInputGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const CoordinateLabel = styled.label`
  color: #fff;
  font-size: 14px;
  font-weight: 600;
`;

const CoordinateInput = styled.input`
  padding: 14px 18px;
  background: rgba(255, 255, 255, 0.1);
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 10px;
  color: #fff;
  font-size: 15px;
  transition: all 0.3s;

  &::placeholder {
    color: rgba(255, 255, 255, 0.5);
  }

  &:focus {
    outline: none;
    border-color: #667eea;
    background: rgba(255, 255, 255, 0.15);
  }
`;

const HelpText = styled.div`
  margin-top: 20px;
  padding: 12px;
  background: rgba(102, 126, 234, 0.1);
  border: 1px solid rgba(102, 126, 234, 0.3);
  border-radius: 8px;
  color: rgba(255, 255, 255, 0.7);
  font-size: 13px;
  text-align: center;
`;

export default ManualLocationSearch;
