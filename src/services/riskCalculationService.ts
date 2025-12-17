/**
 * GLOBAL RISK CALCULATION SERVICE
 * Calculates dynamic risk scores based on weather, alerts, and location data
 */

export interface WeatherRiskFactors {
  temp: number;
  feelsLike: number;
  condition: string;
  humidity: number;
  windSpeed: number;
  pressure: number;
  visibility: number;
  uvIndex: number;
}

export interface AlertRiskData {
  severity: 'critical' | 'high' | 'moderate' | 'low';
  count: number;
}

export class RiskCalculationService {
  /**
   * Calculate weather-based risk score (0-10 scale)
   */
  static calculateWeatherRisk(weather: WeatherRiskFactors): number {
    let risk = 0;

    // Temperature extremes (0-2.5 points)
    if (weather.temp > 40 || weather.temp < -10) risk += 2.5;
    else if (weather.temp > 35 || weather.temp < 0) risk += 1.5;
    else if (weather.temp > 32 || weather.temp < 5) risk += 0.8;

    // Feels-like temperature (0-1.5 points)
    const heatIndex = Math.abs(weather.temp - weather.feelsLike);
    if (heatIndex > 10) risk += 1.5;
    else if (heatIndex > 5) risk += 0.8;

    // Weather conditions (0-3 points)
    const dangerousConditions = ['Thunderstorm', 'Tornado', 'Hurricane', 'Squall'];
    const severeConditions = ['Snow', 'Rain', 'Drizzle', 'Mist', 'Fog', 'Hail'];
    
    if (dangerousConditions.some(cond => weather.condition.includes(cond))) risk += 3;
    else if (severeConditions.some(cond => weather.condition.includes(cond))) risk += 1.5;
    else if (weather.condition.includes('Clouds')) risk += 0.3;

    // Wind speed (0-1.5 points) - km/h
    if (weather.windSpeed > 70) risk += 1.5; // Severe gale
    else if (weather.windSpeed > 50) risk += 1; // Strong wind
    else if (weather.windSpeed > 30) risk += 0.5;

    // Humidity extremes (0-1 point)
    if (weather.humidity > 85 || weather.humidity < 20) risk += 1;
    else if (weather.humidity > 75 || weather.humidity < 30) risk += 0.5;

    // Low visibility (0-0.8 points) - km
    if (weather.visibility < 1) risk += 0.8;
    else if (weather.visibility < 3) risk += 0.5;
    else if (weather.visibility < 5) risk += 0.2;

    // UV Index (0-0.7 points)
    if (weather.uvIndex > 10) risk += 0.7;
    else if (weather.uvIndex > 7) risk += 0.4;

    return Math.min(Math.round(risk * 10) / 10, 10); // Cap at 10, round to 1 decimal
  }

  /**
   * Calculate alert-based risk score (0-10 scale)
   */
  static calculateAlertRisk(alerts: AlertRiskData[]): number {
    let risk = 0;

    alerts.forEach(alert => {
      const severityMultiplier = {
        critical: 3,
        high: 2,
        moderate: 1.2,
        low: 0.5
      }[alert.severity];

      risk += severityMultiplier * Math.min(alert.count, 5); // Cap count effect at 5
    });

    return Math.min(Math.round(risk * 10) / 10, 10);
  }

  /**
   * Calculate combined global risk score (0-10 scale)
   * Weighted average: 60% weather + 40% alerts
   */
  static calculateGlobalRisk(
    weatherRisk: number, 
    alertRisk: number
  ): number {
    const weightedRisk = (weatherRisk * 0.6) + (alertRisk * 0.4);
    return Math.min(Math.round(weightedRisk * 10) / 10, 10);
  }

  /**
   * Get risk level classification
   */
  static getRiskLevel(risk: number): 'critical' | 'high' | 'moderate' | 'low' {
    if (risk >= 8) return 'critical';
    if (risk >= 6) return 'high';
    if (risk >= 3) return 'moderate';
    return 'low';
  }

  /**
   * Get risk color based on score
   */
  static getRiskColor(risk: number): string {
    if (risk >= 7) return '#EF4444'; // Red
    if (risk >= 4) return '#F59E0B'; // Yellow/Orange
    return '#22C55E'; // Green
  }
}

export default RiskCalculationService;
