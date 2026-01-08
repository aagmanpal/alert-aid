/**
 * Traffic-Aware Routing Service
 * Real-time traffic integration for evacuation and emergency routing
 */

// Traffic condition levels
type TrafficLevel = 'free_flow' | 'light' | 'moderate' | 'heavy' | 'standstill' | 'blocked';

// Road segment with traffic
interface RoadSegment {
  id: string;
  name: string;
  start: { lat: number; lng: number };
  end: { lat: number; lng: number };
  length: number; // km
  baseSpeed: number; // km/h
  currentSpeed: number; // km/h
  trafficLevel: TrafficLevel;
  roadType: RoadType;
  lanes: number;
  isOneWay: boolean;
  incidents: TrafficIncident[];
  lastUpdated: Date;
}

// Road types
type RoadType = 'highway' | 'national' | 'state' | 'city' | 'local' | 'service';

// Traffic incident
interface TrafficIncident {
  id: string;
  type: IncidentType;
  severity: 'minor' | 'moderate' | 'severe' | 'critical';
  location: { lat: number; lng: number };
  description: string;
  affectedLanes: number;
  startTime: Date;
  expectedClearTime?: Date;
  verified: boolean;
  source: 'official' | 'crowd' | 'sensor' | 'camera';
}

// Incident types
type IncidentType = 
  | 'accident'
  | 'breakdown'
  | 'construction'
  | 'flooding'
  | 'landslide'
  | 'protest'
  | 'event'
  | 'road_closure'
  | 'fallen_tree'
  | 'fire'
  | 'police_activity';

// Route with traffic
interface TrafficRoute {
  id: string;
  segments: RoadSegment[];
  totalDistance: number; // km
  baseTime: number; // minutes without traffic
  currentTime: number; // minutes with traffic
  delay: number; // minutes
  trafficScore: number; // 0-100 (100 = free flow)
  incidents: TrafficIncident[];
  alternativeRoutes: AlternativeRoute[];
  waypoints: { lat: number; lng: number }[];
  instructions: RouteInstruction[];
}

// Alternative route
interface AlternativeRoute {
  id: string;
  name: string;
  distance: number;
  time: number;
  delay: number;
  trafficScore: number;
  description: string;
  recommended: boolean;
}

// Route instruction
interface RouteInstruction {
  text: string;
  distance: number;
  duration: number;
  maneuver: ManeuverType;
  roadName: string;
  trafficInfo?: string;
  coordinates: { lat: number; lng: number };
}

// Maneuver types
type ManeuverType = 
  | 'depart'
  | 'arrive'
  | 'turn_left'
  | 'turn_right'
  | 'slight_left'
  | 'slight_right'
  | 'sharp_left'
  | 'sharp_right'
  | 'uturn'
  | 'straight'
  | 'merge'
  | 'exit'
  | 'roundabout'
  | 'ferry';

// Traffic flow data
interface TrafficFlowData {
  timestamp: Date;
  segments: {
    segmentId: string;
    speed: number;
    freeFlowSpeed: number;
    confidence: number;
    travelTime: number;
  }[];
}

// Traffic prediction
interface TrafficPrediction {
  segmentId: string;
  predictions: {
    time: Date;
    expectedSpeed: number;
    confidence: number;
    trafficLevel: TrafficLevel;
  }[];
}

// Traffic colors
const TRAFFIC_COLORS: Record<TrafficLevel, string> = {
  free_flow: '#4CAF50',
  light: '#8BC34A',
  moderate: '#FFC107',
  heavy: '#FF9800',
  standstill: '#F44336',
  blocked: '#212121',
};

// Speed factors by traffic level
const SPEED_FACTORS: Record<TrafficLevel, number> = {
  free_flow: 1.0,
  light: 0.85,
  moderate: 0.6,
  heavy: 0.35,
  standstill: 0.1,
  blocked: 0,
};

// Base speeds by road type (km/h)
const BASE_SPEEDS: Record<RoadType, number> = {
  highway: 100,
  national: 80,
  state: 60,
  city: 40,
  local: 30,
  service: 20,
};

// Sample road network
const SAMPLE_ROAD_NETWORK: Omit<RoadSegment, 'currentSpeed' | 'trafficLevel' | 'incidents' | 'lastUpdated'>[] = [
  // Mumbai road segments
  { id: 'seg-1', name: 'Western Express Highway', start: { lat: 19.1176, lng: 72.8562 }, end: { lat: 19.0600, lng: 72.8400 }, length: 8.5, baseSpeed: 80, roadType: 'highway', lanes: 6, isOneWay: false },
  { id: 'seg-2', name: 'Eastern Express Highway', start: { lat: 19.0760, lng: 72.8777 }, end: { lat: 19.0200, lng: 72.8500 }, length: 7.2, baseSpeed: 80, roadType: 'highway', lanes: 6, isOneWay: false },
  { id: 'seg-3', name: 'Sion-Panvel Highway', start: { lat: 19.0440, lng: 72.8697 }, end: { lat: 19.0100, lng: 73.0500 }, length: 22, baseSpeed: 60, roadType: 'national', lanes: 4, isOneWay: false },
  { id: 'seg-4', name: 'LBS Marg', start: { lat: 19.0596, lng: 72.8861 }, end: { lat: 19.0200, lng: 72.8600 }, length: 5.8, baseSpeed: 40, roadType: 'city', lanes: 4, isOneWay: false },
  { id: 'seg-5', name: 'SV Road', start: { lat: 19.1196, lng: 72.8489 }, end: { lat: 19.0800, lng: 72.8300 }, length: 6.2, baseSpeed: 35, roadType: 'city', lanes: 4, isOneWay: false },
  // Delhi road segments
  { id: 'seg-6', name: 'NH44 (Delhi-Gurgaon)', start: { lat: 28.5355, lng: 77.2090 }, end: { lat: 28.4595, lng: 77.0266 }, length: 25, baseSpeed: 100, roadType: 'highway', lanes: 8, isOneWay: false },
  { id: 'seg-7', name: 'Ring Road', start: { lat: 28.6139, lng: 77.2090 }, end: { lat: 28.5800, lng: 77.1800 }, length: 8, baseSpeed: 60, roadType: 'city', lanes: 6, isOneWay: false },
  { id: 'seg-8', name: 'Outer Ring Road', start: { lat: 28.6500, lng: 77.2200 }, end: { lat: 28.6000, lng: 77.2800 }, length: 12, baseSpeed: 70, roadType: 'city', lanes: 6, isOneWay: false },
  // Chennai segments
  { id: 'seg-9', name: 'OMR (Old Mahabalipuram Road)', start: { lat: 12.9716, lng: 80.2445 }, end: { lat: 12.8400, lng: 80.2200 }, length: 18, baseSpeed: 60, roadType: 'state', lanes: 6, isOneWay: false },
  { id: 'seg-10', name: 'Mount Road', start: { lat: 13.0827, lng: 80.2707 }, end: { lat: 13.0400, lng: 80.2500 }, length: 6, baseSpeed: 35, roadType: 'city', lanes: 4, isOneWay: false },
];

// Sample incidents
const SAMPLE_INCIDENTS: TrafficIncident[] = [
  {
    id: 'inc-1',
    type: 'flooding',
    severity: 'severe',
    location: { lat: 19.0760, lng: 72.8777 },
    description: 'Heavy waterlogging near Sion',
    affectedLanes: 2,
    startTime: new Date(Date.now() - 2 * 60 * 60 * 1000),
    verified: true,
    source: 'official',
  },
  {
    id: 'inc-2',
    type: 'accident',
    severity: 'moderate',
    location: { lat: 28.5355, lng: 77.2090 },
    description: 'Two vehicle collision on NH44',
    affectedLanes: 1,
    startTime: new Date(Date.now() - 30 * 60 * 1000),
    expectedClearTime: new Date(Date.now() + 60 * 60 * 1000),
    verified: true,
    source: 'crowd',
  },
  {
    id: 'inc-3',
    type: 'construction',
    severity: 'minor',
    location: { lat: 12.9716, lng: 80.2445 },
    description: 'Metro construction work',
    affectedLanes: 1,
    startTime: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    verified: true,
    source: 'official',
  },
];

class TrafficRoutingService {
  private static instance: TrafficRoutingService;
  private roadSegments: Map<string, RoadSegment> = new Map();
  private incidents: Map<string, TrafficIncident> = new Map();
  private trafficUpdateInterval: NodeJS.Timeout | null = null;
  private listeners: ((segments: RoadSegment[]) => void)[] = [];

  private constructor() {
    this.initializeData();
    this.startTrafficUpdates();
  }

  public static getInstance(): TrafficRoutingService {
    if (!TrafficRoutingService.instance) {
      TrafficRoutingService.instance = new TrafficRoutingService();
    }
    return TrafficRoutingService.instance;
  }

  /**
   * Initialize with sample data
   */
  private initializeData(): void {
    SAMPLE_ROAD_NETWORK.forEach((segment) => {
      const fullSegment: RoadSegment = {
        ...segment,
        currentSpeed: segment.baseSpeed,
        trafficLevel: 'free_flow',
        incidents: [],
        lastUpdated: new Date(),
      };
      this.roadSegments.set(segment.id, fullSegment);
    });

    SAMPLE_INCIDENTS.forEach((incident) => {
      this.incidents.set(incident.id, incident);
      this.applyIncidentToSegments(incident);
    });

    this.simulateTraffic();
  }

  /**
   * Simulate traffic conditions
   */
  private simulateTraffic(): void {
    this.roadSegments.forEach((segment) => {
      // Simulate based on time of day
      const hour = new Date().getHours();
      const isPeakHour = (hour >= 8 && hour <= 10) || (hour >= 17 && hour <= 20);
      
      let trafficFactor = isPeakHour ? 0.4 + Math.random() * 0.3 : 0.7 + Math.random() * 0.3;
      
      // Weather factor (simulated)
      const weatherFactor = 0.9 + Math.random() * 0.1;
      
      // Road type factor
      const roadTypeFactor = segment.roadType === 'highway' ? 1 : 
                            segment.roadType === 'national' ? 0.95 : 
                            segment.roadType === 'city' ? 0.8 : 0.9;

      const speedRatio = trafficFactor * weatherFactor * roadTypeFactor;
      segment.currentSpeed = Math.round(segment.baseSpeed * speedRatio);
      segment.trafficLevel = this.getTrafficLevel(speedRatio);
      segment.lastUpdated = new Date();
    });
  }

  /**
   * Apply incident effects to segments
   */
  private applyIncidentToSegments(incident: TrafficIncident): void {
    const affectedRadius = incident.severity === 'critical' ? 5 : 
                          incident.severity === 'severe' ? 3 : 
                          incident.severity === 'moderate' ? 1.5 : 0.5;

    this.roadSegments.forEach((segment) => {
      const distance = this.calculateDistance(
        incident.location.lat, incident.location.lng,
        (segment.start.lat + segment.end.lat) / 2,
        (segment.start.lng + segment.end.lng) / 2
      );

      if (distance <= affectedRadius) {
        segment.incidents.push(incident);
        
        const reductionFactor = incident.severity === 'critical' ? 0.1 :
                               incident.severity === 'severe' ? 0.3 :
                               incident.severity === 'moderate' ? 0.6 : 0.8;
        
        segment.currentSpeed = Math.round(segment.currentSpeed * reductionFactor);
        segment.trafficLevel = this.getTrafficLevel(segment.currentSpeed / segment.baseSpeed);
      }
    });
  }

  /**
   * Get traffic level from speed ratio
   */
  private getTrafficLevel(ratio: number): TrafficLevel {
    if (ratio >= 0.9) return 'free_flow';
    if (ratio >= 0.7) return 'light';
    if (ratio >= 0.5) return 'moderate';
    if (ratio >= 0.25) return 'heavy';
    if (ratio > 0) return 'standstill';
    return 'blocked';
  }

  /**
   * Start periodic traffic updates
   */
  private startTrafficUpdates(): void {
    this.trafficUpdateInterval = setInterval(() => {
      this.simulateTraffic();
      this.notifyListeners();
    }, 60000); // Update every minute
  }

  /**
   * Stop traffic updates
   */
  public stopTrafficUpdates(): void {
    if (this.trafficUpdateInterval) {
      clearInterval(this.trafficUpdateInterval);
      this.trafficUpdateInterval = null;
    }
  }

  /**
   * Calculate route with traffic
   */
  public calculateRoute(
    origin: { lat: number; lng: number },
    destination: { lat: number; lng: number },
    options: {
      avoidTolls?: boolean;
      avoidHighways?: boolean;
      preferFastestRoute?: boolean;
    } = {}
  ): TrafficRoute {
    // Find segments between origin and destination
    const relevantSegments = this.findRelevantSegments(origin, destination);
    
    // Calculate route metrics
    let totalDistance = 0;
    let baseTime = 0;
    let currentTime = 0;
    const routeIncidents: TrafficIncident[] = [];
    const waypoints: { lat: number; lng: number }[] = [origin];

    relevantSegments.forEach((segment) => {
      totalDistance += segment.length;
      baseTime += (segment.length / segment.baseSpeed) * 60; // minutes
      currentTime += (segment.length / segment.currentSpeed) * 60; // minutes
      routeIncidents.push(...segment.incidents);
      waypoints.push(segment.end);
    });

    waypoints.push(destination);

    // Calculate traffic score
    const trafficScore = baseTime > 0 ? Math.round((baseTime / currentTime) * 100) : 100;

    // Generate instructions
    const instructions = this.generateInstructions(relevantSegments, origin, destination);

    // Find alternative routes
    const alternativeRoutes = this.findAlternativeRoutes(origin, destination, relevantSegments);

    return {
      id: `route-${Date.now()}`,
      segments: relevantSegments,
      totalDistance: Math.round(totalDistance * 10) / 10,
      baseTime: Math.round(baseTime),
      currentTime: Math.round(currentTime),
      delay: Math.round(currentTime - baseTime),
      trafficScore,
      incidents: routeIncidents,
      alternativeRoutes,
      waypoints,
      instructions,
    };
  }

  /**
   * Find relevant segments
   */
  private findRelevantSegments(
    origin: { lat: number; lng: number },
    destination: { lat: number; lng: number }
  ): RoadSegment[] {
    const segments = Array.from(this.roadSegments.values());
    
    // Sort by proximity to route line
    return segments
      .filter((segment) => {
        const segmentCenter = {
          lat: (segment.start.lat + segment.end.lat) / 2,
          lng: (segment.start.lng + segment.end.lng) / 2,
        };
        const routeCenter = {
          lat: (origin.lat + destination.lat) / 2,
          lng: (origin.lng + destination.lng) / 2,
        };
        const distance = this.calculateDistance(
          segmentCenter.lat, segmentCenter.lng,
          routeCenter.lat, routeCenter.lng
        );
        return distance < 50; // Within 50km of route center
      })
      .slice(0, 5); // Limit to 5 segments for demo
  }

  /**
   * Generate route instructions
   */
  private generateInstructions(
    segments: RoadSegment[],
    origin: { lat: number; lng: number },
    destination: { lat: number; lng: number }
  ): RouteInstruction[] {
    const instructions: RouteInstruction[] = [];

    // Start instruction
    instructions.push({
      text: `Start from your location`,
      distance: 0,
      duration: 0,
      maneuver: 'depart',
      roadName: '',
      coordinates: origin,
    });

    // Segment instructions
    segments.forEach((segment, index) => {
      const duration = (segment.length / segment.currentSpeed) * 60;
      const trafficInfo = segment.trafficLevel !== 'free_flow' 
        ? `Traffic: ${segment.trafficLevel.replace('_', ' ')}`
        : undefined;

      instructions.push({
        text: `Continue on ${segment.name} for ${segment.length.toFixed(1)} km`,
        distance: segment.length,
        duration,
        maneuver: index === 0 ? 'straight' : this.getManeuver(segments[index - 1], segment),
        roadName: segment.name,
        trafficInfo,
        coordinates: segment.start,
      });

      // Add incident warnings
      segment.incidents.forEach((incident) => {
        instructions.push({
          text: `⚠️ ${incident.type.replace('_', ' ')}: ${incident.description}`,
          distance: 0,
          duration: 0,
          maneuver: 'straight',
          roadName: segment.name,
          coordinates: incident.location,
        });
      });
    });

    // End instruction
    instructions.push({
      text: `Arrive at destination`,
      distance: 0,
      duration: 0,
      maneuver: 'arrive',
      roadName: '',
      coordinates: destination,
    });

    return instructions;
  }

  /**
   * Get maneuver type between segments
   */
  private getManeuver(prev: RoadSegment, current: RoadSegment): ManeuverType {
    const prevBearing = this.calculateBearing(prev.start, prev.end);
    const currentBearing = this.calculateBearing(current.start, current.end);
    const turn = (currentBearing - prevBearing + 360) % 360;

    if (turn < 30 || turn > 330) return 'straight';
    if (turn >= 30 && turn < 60) return 'slight_right';
    if (turn >= 60 && turn < 120) return 'turn_right';
    if (turn >= 120 && turn < 150) return 'sharp_right';
    if (turn >= 150 && turn < 210) return 'uturn';
    if (turn >= 210 && turn < 240) return 'sharp_left';
    if (turn >= 240 && turn < 300) return 'turn_left';
    return 'slight_left';
  }

  /**
   * Calculate bearing between two points
   */
  private calculateBearing(
    from: { lat: number; lng: number },
    to: { lat: number; lng: number }
  ): number {
    const lat1 = this.toRadians(from.lat);
    const lat2 = this.toRadians(to.lat);
    const dLng = this.toRadians(to.lng - from.lng);

    const x = Math.sin(dLng) * Math.cos(lat2);
    const y = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);

    const bearing = Math.atan2(x, y);
    return (this.toDegrees(bearing) + 360) % 360;
  }

  /**
   * Find alternative routes
   */
  private findAlternativeRoutes(
    origin: { lat: number; lng: number },
    destination: { lat: number; lng: number },
    mainRoute: RoadSegment[]
  ): AlternativeRoute[] {
    const alternatives: AlternativeRoute[] = [];

    // Highway route
    const highwaySegments = Array.from(this.roadSegments.values())
      .filter((s) => s.roadType === 'highway');
    
    if (highwaySegments.length > 0) {
      const hwDistance = highwaySegments.reduce((sum, s) => sum + s.length, 0);
      const hwTime = highwaySegments.reduce((sum, s) => sum + (s.length / s.currentSpeed) * 60, 0);
      
      alternatives.push({
        id: 'alt-highway',
        name: 'Via Highway',
        distance: Math.round(hwDistance * 1.1 * 10) / 10,
        time: Math.round(hwTime * 0.9),
        delay: 5,
        trafficScore: 75,
        description: 'Faster route via highway with moderate traffic',
        recommended: hwTime < (mainRoute.reduce((sum, s) => sum + (s.length / s.currentSpeed) * 60, 0)),
      });
    }

    // Local route
    const localSegments = Array.from(this.roadSegments.values())
      .filter((s) => s.roadType === 'local' || s.roadType === 'city');
    
    if (localSegments.length > 0) {
      alternatives.push({
        id: 'alt-local',
        name: 'Via Local Roads',
        distance: Math.round(localSegments.reduce((sum, s) => sum + s.length, 0) * 1.3 * 10) / 10,
        time: Math.round(localSegments.reduce((sum, s) => sum + (s.length / s.currentSpeed) * 60, 0) * 1.2),
        delay: 2,
        trafficScore: 85,
        description: 'Longer route through local roads with less traffic',
        recommended: false,
      });
    }

    return alternatives;
  }

  /**
   * Get traffic on segment
   */
  public getSegmentTraffic(segmentId: string): RoadSegment | undefined {
    return this.roadSegments.get(segmentId);
  }

  /**
   * Get all segments
   */
  public getAllSegments(): RoadSegment[] {
    return Array.from(this.roadSegments.values());
  }

  /**
   * Get segments in bounds
   */
  public getSegmentsInBounds(bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  }): RoadSegment[] {
    return Array.from(this.roadSegments.values()).filter((segment) => {
      const lat = (segment.start.lat + segment.end.lat) / 2;
      const lng = (segment.start.lng + segment.end.lng) / 2;
      return lat >= bounds.south && lat <= bounds.north &&
             lng >= bounds.west && lng <= bounds.east;
    });
  }

  /**
   * Report incident
   */
  public reportIncident(
    type: IncidentType,
    location: { lat: number; lng: number },
    description: string,
    severity: TrafficIncident['severity'] = 'moderate'
  ): TrafficIncident {
    const incident: TrafficIncident = {
      id: `inc-${Date.now()}`,
      type,
      severity,
      location,
      description,
      affectedLanes: 1,
      startTime: new Date(),
      verified: false,
      source: 'crowd',
    };

    this.incidents.set(incident.id, incident);
    this.applyIncidentToSegments(incident);
    this.notifyListeners();

    return incident;
  }

  /**
   * Clear incident
   */
  public clearIncident(incidentId: string): void {
    const incident = this.incidents.get(incidentId);
    if (!incident) return;

    this.incidents.delete(incidentId);

    // Remove from segments and recalculate
    this.roadSegments.forEach((segment) => {
      segment.incidents = segment.incidents.filter((i) => i.id !== incidentId);
    });

    this.simulateTraffic();
    this.notifyListeners();
  }

  /**
   * Get active incidents
   */
  public getActiveIncidents(): TrafficIncident[] {
    return Array.from(this.incidents.values());
  }

  /**
   * Get incidents near location
   */
  public getIncidentsNear(
    lat: number,
    lng: number,
    radiusKm: number = 10
  ): TrafficIncident[] {
    return Array.from(this.incidents.values()).filter((incident) => {
      const distance = this.calculateDistance(lat, lng, incident.location.lat, incident.location.lng);
      return distance <= radiusKm;
    });
  }

  /**
   * Predict traffic
   */
  public predictTraffic(segmentId: string, hoursAhead: number = 1): TrafficPrediction {
    const segment = this.roadSegments.get(segmentId);
    const predictions: TrafficPrediction['predictions'] = [];

    for (let h = 1; h <= hoursAhead; h++) {
      const futureTime = new Date(Date.now() + h * 60 * 60 * 1000);
      const hour = futureTime.getHours();
      const isPeakHour = (hour >= 8 && hour <= 10) || (hour >= 17 && hour <= 20);

      const expectedRatio = isPeakHour ? 0.5 : 0.8;
      const expectedSpeed = segment ? segment.baseSpeed * expectedRatio : 40;

      predictions.push({
        time: futureTime,
        expectedSpeed: Math.round(expectedSpeed),
        confidence: 0.7 - (h * 0.1),
        trafficLevel: this.getTrafficLevel(expectedRatio),
      });
    }

    return {
      segmentId,
      predictions,
    };
  }

  /**
   * Get traffic color
   */
  public getTrafficColor(level: TrafficLevel): string {
    return TRAFFIC_COLORS[level];
  }

  /**
   * Calculate ETA
   */
  public calculateETA(route: TrafficRoute): Date {
    return new Date(Date.now() + route.currentTime * 60 * 1000);
  }

  /**
   * Subscribe to updates
   */
  public subscribe(callback: (segments: RoadSegment[]) => void): () => void {
    this.listeners.push(callback);
    return () => {
      const index = this.listeners.indexOf(callback);
      if (index > -1) this.listeners.splice(index, 1);
    };
  }

  /**
   * Notify listeners
   */
  private notifyListeners(): void {
    const segments = Array.from(this.roadSegments.values());
    this.listeners.forEach((callback) => callback(segments));
  }

  /**
   * Helper: Calculate distance
   */
  private calculateDistance(lat1: number, lng1: number, lat2: number, lng2: number): number {
    const R = 6371;
    const dLat = this.toRadians(lat2 - lat1);
    const dLng = this.toRadians(lng2 - lng1);
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(this.toRadians(lat1)) * Math.cos(this.toRadians(lat2)) *
              Math.sin(dLng / 2) * Math.sin(dLng / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  private toRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  private toDegrees(radians: number): number {
    return radians * (180 / Math.PI);
  }

  /**
   * Format duration
   */
  public formatDuration(minutes: number): string {
    if (minutes < 1) return '< 1 min';
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return mins > 0 ? `${hours} hr ${mins} min` : `${hours} hr`;
  }
}

export const trafficRoutingService = TrafficRoutingService.getInstance();
export type {
  RoadSegment,
  RoadType,
  TrafficLevel,
  TrafficIncident,
  IncidentType,
  TrafficRoute,
  AlternativeRoute,
  RouteInstruction,
  ManeuverType,
  TrafficFlowData,
  TrafficPrediction,
};
