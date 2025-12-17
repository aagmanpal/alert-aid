import React from 'react';
import { Cloud, Thermometer, Droplets, Wind, Eye, Gauge } from 'lucide-react';
import { Card, CardHeader, CardContent, MetricValue, Badge, IconWrapper, Grid, FlexRow } from './CardStyles';

interface WeatherCardProps {
  weather: any;
}

const WeatherCard: React.FC<WeatherCardProps> = ({ weather }) => {
  if (!weather) {
    return (
      <Card variant="info" size="medium">
        <CardHeader>
          <h3>
            <IconWrapper variant="info">
              <Cloud size={20} />
            </IconWrapper>
            Weather Conditions
          </h3>
          <Badge variant="warning">Loading...</Badge>
        </CardHeader>
        <CardContent>
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <p style={{ color: '#a0aec0' }}>Loading weather data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getWeatherIcon = () => {
    if (!weather?.weather_icon) return '🌤️';
    const icon = weather.weather_icon;
    const iconMap: { [key: string]: string } = {
      '01d': '☀️', '01n': '🌙',
      '02d': '⛅', '02n': '🌙',
      '03d': '☁️', '03n': '☁️',
      '04d': '☁️', '04n': '☁️',
      '09d': '🌧️', '09n': '🌧️',
      '10d': '🌦️', '10n': '🌧️',
      '11d': '⛈️', '11n': '⛈️',
      '13d': '🌨️', '13n': '🌨️',
      '50d': '🌫️', '50n': '🌫️'
    };
    return iconMap[icon] || '🌤️';
  };

  const getTemperatureColor = (temp: number) => {
    if (temp >= 35) return '#f56565';
    if (temp >= 25) return '#fbd38d';
    if (temp >= 15) return '#68d391';
    if (temp >= 5) return '#4fd1c7';
    return '#9f7aea';
  };

  const getHumidityVariant = (humidity: number) => {
    if (humidity >= 80) return 'warning';
    if (humidity >= 60) return 'info';
    return 'success';
  };

  const getWindVariant = (speed: number) => {
    if (speed >= 25) return 'danger';
    if (speed >= 15) return 'warning';
    return 'success';
  };

  return (
    <Card variant="info" size="medium">
      <CardHeader>
        <h3>
          <IconWrapper variant="info">
            <Cloud size={20} />
          </IconWrapper>
          Weather Conditions
        </h3>
        <Badge variant={weather.is_real ? 'success' : 'warning'}>
          {weather.is_real ? 'Live Data' : 'Simulated'}
        </Badge>
      </CardHeader>
      
      <CardContent>
        {/* Main Temperature Display */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ fontSize: '4rem', marginBottom: '0.5rem' }}>
            {getWeatherIcon()}
          </div>
          <MetricValue 
            size="large" 
            style={{ color: getTemperatureColor(weather.temperature || 0) }}
          >
            {weather.temperature || '--'}°C
          </MetricValue>
          <p style={{ margin: '0.5rem 0', fontSize: '1rem', color: '#cbd5e0', textTransform: 'capitalize' }}>
            {weather.conditions || 'Loading...'}
          </p>
          {weather.temperature_feels_like && (
            <p style={{ margin: '0', fontSize: '0.9rem', color: '#a0aec0' }}>
              Feels like {weather.temperature_feels_like}°C
            </p>
          )}
        </div>

        {/* Weather Details Grid */}
        <Grid columns={2} gap="1rem">
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.75rem',
            padding: '1rem',
            background: 'rgba(76, 209, 199, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(76, 209, 199, 0.2)'
          }}>
            <Droplets size={18} color="#4fd1c7" />
            <div>
              <MetricValue size="small" style={{ fontSize: '1.2rem' }}>
                {weather.humidity || '--'}%
              </MetricValue>
              <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>Humidity</div>
            </div>
          </div>

          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.75rem',
            padding: '1rem',
            background: 'rgba(251, 191, 36, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(251, 191, 36, 0.2)'
          }}>
            <Wind size={18} color="#fbd38d" />
            <div>
              <MetricValue size="small" style={{ fontSize: '1.2rem' }}>
                {weather.wind_speed || '--'}
              </MetricValue>
              <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>km/h</div>
            </div>
          </div>

          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.75rem',
            padding: '1rem',
            background: 'rgba(168, 85, 247, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(168, 85, 247, 0.2)'
          }}>
            <Gauge size={18} color="#a855f7" />
            <div>
              <MetricValue size="small" style={{ fontSize: '1.2rem' }}>
                {weather.pressure || '--'}
              </MetricValue>
              <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>hPa</div>
            </div>
          </div>

          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.75rem',
            padding: '1rem',
            background: 'rgba(72, 187, 120, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(72, 187, 120, 0.2)'
          }}>
            <Eye size={18} color="#48bb78" />
            <div>
              <MetricValue size="small" style={{ fontSize: '1.2rem' }}>
                {weather.visibility || '--'}
              </MetricValue>
              <div style={{ fontSize: '0.8rem', color: '#a0aec0' }}>km</div>
            </div>
          </div>
        </Grid>

        {/* Additional Info */}
        <div style={{ marginTop: '1.5rem' }}>
          <FlexRow gap="0.5rem" justify="space-between" style={{ marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.9rem', color: '#cbd5e0' }}>Wind Direction:</span>
            <span style={{ fontSize: '0.9rem', color: '#fbd38d', fontWeight: '600' }}>
              {weather.wind_direction || '--'}°
            </span>
          </FlexRow>
          
          <FlexRow gap="0.5rem" justify="space-between" style={{ marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.9rem', color: '#cbd5e0' }}>Source:</span>
            <span style={{ fontSize: '0.9rem', color: '#4fd1c7', fontWeight: '600' }}>
              {weather.source || 'OpenWeatherMap'}
            </span>
          </FlexRow>
          
          {weather.last_updated && (
            <FlexRow gap="0.5rem" justify="space-between">
              <span style={{ fontSize: '0.9rem', color: '#cbd5e0' }}>Last Updated:</span>
              <span style={{ fontSize: '0.9rem', color: '#a0aec0' }}>
                {new Date(weather.last_updated).toLocaleTimeString()}
              </span>
            </FlexRow>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default WeatherCard;