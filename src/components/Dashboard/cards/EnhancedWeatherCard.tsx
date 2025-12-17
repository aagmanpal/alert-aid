import React from 'react';
import { Cloud, CloudRain, Sun, CloudSnow, Wind, Eye, Gauge, Droplets, Thermometer, Zap } from 'lucide-react';
import { Card, CardHeader, CardContent, Badge, IconWrapper, MetricValue } from './CardStyles';

interface WeatherCardProps {
  weather: any;
}

const EnhancedWeatherCard: React.FC<WeatherCardProps> = ({ weather }) => {
  const getWeatherIcon = (condition: string, iconCode?: string) => {
    if (iconCode) {
      // Use OpenWeatherMap icon mapping
      const iconMap: { [key: string]: React.ReactNode } = {
        '01d': <Sun size={24} color="#FDB402" />, // clear sky day
        '01n': <Sun size={24} color="#4A5568" />, // clear sky night
        '02d': <Cloud size={24} color="#A0AEC0" />, // few clouds day
        '02n': <Cloud size={24} color="#4A5568" />, // few clouds night
        '03d': <Cloud size={24} color="#A0AEC0" />, // scattered clouds
        '03n': <Cloud size={24} color="#4A5568" />,
        '04d': <Cloud size={24} color="#718096" />, // broken clouds
        '04n': <Cloud size={24} color="#4A5568" />,
        '09d': <CloudRain size={24} color="#4299E1" />, // shower rain
        '09n': <CloudRain size={24} color="#2B6CB0" />,
        '10d': <CloudRain size={24} color="#4299E1" />, // rain
        '10n': <CloudRain size={24} color="#2B6CB0" />,
        '11d': <Zap size={24} color="#805AD5" />, // thunderstorm
        '11n': <Zap size={24} color="#553C9A" />,
        '13d': <CloudSnow size={24} color="#E2E8F0" />, // snow
        '13n': <CloudSnow size={24} color="#A0AEC0" />,
        '50d': <Wind size={24} color="#A0AEC0" />, // mist
        '50n': <Wind size={24} color="#718096" />
      };
      
      return iconMap[iconCode] || <Cloud size={24} color="#A0AEC0" />;
    }
    
    // Fallback to condition-based mapping
    const conditionLower = condition?.toLowerCase() || '';
    if (conditionLower.includes('rain') || conditionLower.includes('shower')) {
      return <CloudRain size={24} color="#4299E1" />;
    } else if (conditionLower.includes('cloud')) {
      return <Cloud size={24} color="#A0AEC0" />;
    } else if (conditionLower.includes('clear') || conditionLower.includes('sun')) {
      return <Sun size={24} color="#FDB402" />;
    } else if (conditionLower.includes('snow')) {
      return <CloudSnow size={24} color="#E2E8F0" />;
    } else if (conditionLower.includes('thunder')) {
      return <Zap size={24} color="#805AD5" />;
    } else if (conditionLower.includes('wind') || conditionLower.includes('mist')) {
      return <Wind size={24} color="#A0AEC0" />;
    }
    
    return <Cloud size={24} color="#A0AEC0" />;
  };

  const getTemperatureColor = (temp: number) => {
    if (temp >= 30) return '#F56565'; // Hot - red
    if (temp >= 20) return '#F6AD55'; // Warm - orange
    if (temp >= 10) return '#4FD1C7'; // Mild - teal
    if (temp >= 0) return '#63B3ED'; // Cool - blue
    return '#B794F6'; // Cold - purple
  };

  const formatTemperature = (temp: number) => {
    return `${Math.round(temp)}°C`;
  };

  const getDataSourceBadge = () => {
    if (weather?.is_real === true) {
      return <Badge variant="success">LIVE OpenWeatherMap</Badge>;
    } else if (weather?.cached === true) {
      return <Badge variant="warning">CACHED Data</Badge>;
    } else {
      return <Badge variant="info">SIMULATED Data</Badge>;
    }
  };

  // Show loading state if no weather data
  if (!weather) {
    return (
      <Card variant="info" size="medium">
        <CardHeader>
          <h3>
            <IconWrapper variant="info">
              <Cloud size={20} />
            </IconWrapper>
            Live Weather
          </h3>
          <Badge variant="warning">Loading...</Badge>
        </CardHeader>
        <CardContent>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '200px',
            background: 'linear-gradient(90deg, #f0f0f0 25%, transparent 37%, #f0f0f0 63%)',
            backgroundSize: '400px 100%',
            animation: 'shimmer 1.5s ease-in-out infinite',
            borderRadius: '12px'
          }}>
            <p style={{ color: '#A0AEC0', fontSize: '0.9rem' }}>Fetching weather data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Extract weather data with fallbacks
  const temperature = weather?.temperature ?? weather?.main?.temp ?? 22;
  const feelsLike = weather?.temperature_feels_like ?? weather?.main?.feels_like ?? temperature + 2;
  const humidity = weather?.humidity ?? weather?.main?.humidity ?? 65;
  const windSpeed = weather?.wind_speed ?? weather?.wind?.speed ?? 3.2;
  const pressure = weather?.pressure ?? weather?.main?.pressure ?? 1013;
  const visibility = weather?.visibility ?? 10;
  const conditions = weather?.conditions ?? weather?.weather?.[0]?.description ?? 'Clear sky';
  const iconCode = weather?.weather?.[0]?.icon ?? weather?.weather_icon;
  const lastUpdated = weather?.last_updated ?? new Date().toISOString();

  return (
    <Card variant="info" size="medium" glow={weather?.is_real}>
      <CardHeader>
        <h3>
          <IconWrapper variant="info">
            {getWeatherIcon(conditions, iconCode)}
          </IconWrapper>
          Live Weather
        </h3>
        {getDataSourceBadge()}
      </CardHeader>
      
      <CardContent>
        {/* Main Temperature Display */}
        <div style={{
          textAlign: 'center',
          marginBottom: '2rem',
          padding: '2rem',
          background: weather?.is_real 
            ? 'rgba(79, 209, 199, 0.1)' 
            : 'rgba(246, 173, 85, 0.05)',
          borderRadius: '16px',
          border: `1px solid ${weather?.is_real 
            ? 'rgba(79, 209, 199, 0.2)' 
            : 'rgba(246, 173, 85, 0.1)'}`
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            gap: '1rem',
            marginBottom: '1rem'
          }}>
            {getWeatherIcon(conditions, iconCode)}
            <MetricValue size="large" style={{ color: getTemperatureColor(temperature) }}>
              {formatTemperature(temperature)}
            </MetricValue>
          </div>
          
          <h4 style={{ 
            margin: '0 0 0.5rem 0', 
            fontSize: '1.1rem', 
            color: '#E2E8F0',
            textTransform: 'capitalize'
          }}>
            {conditions}
          </h4>
          
          <p style={{ 
            margin: '0', 
            fontSize: '0.9rem', 
            color: '#A0AEC0' 
          }}>
            Feels like {formatTemperature(feelsLike)}
          </p>
        </div>

        {/* Weather Metrics Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '1rem',
          marginBottom: '1.5rem'
        }}>
          <div style={{
            padding: '1rem',
            background: 'rgba(99, 179, 237, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(99, 179, 237, 0.2)',
            textAlign: 'center'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.5rem' }}>
              <Droplets size={16} color="#63B3ED" />
            </div>
            <MetricValue size="small" style={{ color: '#63B3ED' }}>
              {humidity}%
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#A0AEC0' }}>Humidity</p>
          </div>
          
          <div style={{
            padding: '1rem',
            background: 'rgba(104, 211, 145, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(104, 211, 145, 0.2)',
            textAlign: 'center'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.5rem' }}>
              <Wind size={16} color="#68D391" />
            </div>
            <MetricValue size="small" style={{ color: '#68D391' }}>
              {windSpeed.toFixed(1)}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#A0AEC0' }}>m/s Wind</p>
          </div>
          
          <div style={{
            padding: '1rem',
            background: 'rgba(246, 173, 85, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(246, 173, 85, 0.2)',
            textAlign: 'center'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.5rem' }}>
              <Gauge size={16} color="#F6AD55" />
            </div>
            <MetricValue size="small" style={{ color: '#F6AD55' }}>
              {pressure}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#A0AEC0' }}>hPa Pressure</p>
          </div>
          
          <div style={{
            padding: '1rem',
            background: 'rgba(183, 148, 246, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(183, 148, 246, 0.2)',
            textAlign: 'center'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.5rem' }}>
              <Eye size={16} color="#B794F6" />
            </div>
            <MetricValue size="small" style={{ color: '#B794F6' }}>
              {visibility}
            </MetricValue>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.7rem', color: '#A0AEC0' }}>km Visibility</p>
          </div>
        </div>

        {/* Data Source and Timestamp */}
        <div style={{
          padding: '1rem',
          background: 'rgba(24, 26, 32, 0.8)',
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          fontSize: '0.8rem',
          color: '#A0AEC0'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Data Source:</span>
            <span style={{ 
              color: weather?.is_real ? '#68D391' : '#F6AD55', 
              fontWeight: '600' 
            }}>
              {weather?.source || 'OpenWeatherMap'}
            </span>
          </div>
          
          {lastUpdated && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem' }}>
              <span>Last Updated:</span>
              <span style={{ color: '#4FD1C7' }}>
                {new Date(lastUpdated).toLocaleTimeString()}
              </span>
            </div>
          )}
          
          {weather?.is_real === false && (
            <div style={{ 
              marginTop: '0.5rem',
              padding: '0.5rem',
              background: 'rgba(246, 173, 85, 0.1)',
              borderRadius: '4px',
              fontSize: '0.7rem',
              color: '#F6AD55',
              textAlign: 'center'
            }}>
              ⚠️ Using simulated data - Check internet connection
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default EnhancedWeatherCard;