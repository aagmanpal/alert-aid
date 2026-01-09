/*
 Emergency Dispatch Service (Issue #181)
 Purpose: Computer-Aided Dispatch (CAD)-style orchestration for multi-agency emergency response.
 Scope: Incident intake, triage, unit assignment, live status, ETA/coverage estimates, KPIs, and integration hooks.
*/

// Type aliases and enums
type Agency = 'police' | 'fire' | 'ems' | 'search_rescue' | 'utility' | 'custom';
type UnitCapability =
  | 'basic_life_support'
  | 'advanced_life_support'
  | 'water_rescue'
  | 'high_angle_rescue'
  | 'hazmat'
  | 'k9'
  | 'bomb_squad'
  | 'riot_control'
  | 'traffic_control'
  | 'evacuation'
  | 'wildfire'
  | 'ladder'
  | 'pump'
  | 'investigation'
  | 'communications_relay';

type Priority = 1 | 2 | 3 | 4 | 5; // 1 highest

type IncidentType =
  | 'medical_cardiac_arrest'
  | 'medical_trauma'
  | 'fire_structure'
  | 'fire_wildland'
  | 'mva_injury'
  | 'mva_hazmat'
  | 'crime_in_progress'
  | 'missing_person'
  | 'rescue_water'
  | 'rescue_technical'
  | 'utility_outage'
  | 'hazmat_release'
  | 'evacuation_order'
  | 'other';

type DispatchStatus =
  | 'created'
  | 'queued'
  | 'assigned'
  | 'enroute'
  | 'on_scene'
  | 'transporting'
  | 'at_hospital'
  | 'cleared'
  | 'cancelled';

// Core geospatial types
interface GeoPoint {
  lat: number;
  lon: number;
  accuracy?: number;
}

interface GeoBounds {
  north: number;
  south: number;
  east: number;
  west: number;
}

// Caller and intake
interface CallerInfo {
  name?: string;
  phone?: string;
  language?: string;
  hearingImpaired?: boolean;
  relation?: 'self' | 'bystander' | 'third_party' | 'unknown';
}

interface IntakeMetadata {
  channel: '911' | '112' | 'web' | 'app' | 'radio' | 'sensor' | 'operator';
  receivedAt: Date;
  transcriptId?: string;
  recordingUrl?: string;
}

// Incident and response
interface Incident {
  id: string;
  type: IncidentType;
  subType?: string;
  priority: Priority;
  location: GeoPoint;
  address?: string;
  caller?: CallerInfo;
  description?: string;
  status: DispatchStatus;
  createdAt: Date;
  updatedAt: Date;
  clearedAt?: Date;
  hazards?: string[];
  peopleInvolved?: number;
  injuries?: number;
  fatalities?: number;
  intake: IntakeMetadata;
  assignments: Assignment[];
  history: IncidentEvent[];
  tags?: string[];
  custom?: Record<string, unknown>;
}

interface IncidentEvent {
  id: string;
  type:
    | 'created'
    | 'priority_change'
    | 'status_change'
    | 'assignment_added'
    | 'assignment_removed'
    | 'note'
    | 'location_update'
    | 'hazard_added'
    | 'hazard_removed';
  message: string;
  createdAt: Date;
  actor?: string;
  data?: Record<string, unknown>;
}

// Units and staffing
type UnitStatus =
  | 'available'
  | 'enroute'
  | 'on_scene'
  | 'transporting'
  | 'at_hospital'
  | 'out_of_service'
  | 'maintenance'
  | 'staging'
  | 'training';

interface Unit {
  id: string;
  callsign: string;
  agency: Agency;
  status: UnitStatus;
  location: GeoPoint;
  homeBase?: GeoPoint;
  capabilities: UnitCapability[];
  crewSize: number;
  supervisor?: string;
  lastStatusChange: Date;
  assignedIncidentId?: string;
  etaSeconds?: number;
  fuelLevel?: number;
  batteryLevel?: number;
  equipment?: string[];
  notes?: string[];
}

interface Assignment {
  id: string;
  incidentId: string;
  unitId: string;
  assignedAt: Date;
  acknowledgedAt?: Date;
  status: 'assigned' | 'acknowledged' | 'declined' | 'cancelled' | 'completed';
  etaSeconds?: number;
  routeId?: string;
  role?: 'primary' | 'support' | 'transport' | 'staging' | 'command';
}

// Routing and SLA
interface RoutePlan {
  id: string;
  origin: GeoPoint;
  destination: GeoPoint;
  waypoints?: GeoPoint[];
  distanceMeters: number;
  durationSeconds: number;
  congestionFactor?: number;
  blockedSegments?: number;
  createdAt: Date;
}

interface SLAProfile {
  id: string;
  name: string;
  agency: Agency | 'multi';
  coverageArea: GeoBounds;
  targetResponseSeconds: number; // e.g., medical 480s (8min)
  escalationThresholdSeconds: number; // escalate if exceeded
  priorityOverrides?: { priority: Priority; targetResponseSeconds: number }[];
}

// KPIs and analytics
interface DispatchKPIs {
  totalIncidents: number;
  activeIncidents: number;
  averageTimeToAssign: number;
  averageResponseTime: number;
  onTimeRate: number; // % within SLA
  averageClearTime: number;
  multiAgencyIncidents: number;
  unitUtilization: { agency: Agency; utilization: number }[];
  incidentsByType: { type: IncidentType; count: number }[];
  priorityMix: { priority: Priority; count: number }[];
}

// Integration hooks (non-hard dependency to keep compilation safe)
interface IntegrationHooks {
  notify?: (payload: {
    incident: Incident;
    action: string;
    audience: ('unit' | 'supervisor' | 'dispatcher' | 'public')[];
    message: string;
  }) => Promise<void> | void;

  audit?: (record: {
    category: 'dispatch' | 'assignment' | 'status' | 'kpi';
    action: string;
    entityId: string;
    metadata?: Record<string, unknown>;
    at: Date;
  }) => Promise<void> | void;

  route?: (origin: GeoPoint, destination: GeoPoint, context?: Record<string, unknown>) =>
    Promise<RoutePlan> | RoutePlan;
}

// Sample data
const sampleUnits: Unit[] = [
  {
    id: 'u-ems-1',
    callsign: 'Medic-21',
    agency: 'ems',
    status: 'available',
    location: { lat: 37.7749, lon: -122.4194 },
    homeBase: { lat: 37.77, lon: -122.41 },
    capabilities: ['basic_life_support', 'advanced_life_support'],
    crewSize: 2,
    lastStatusChange: new Date(),
    equipment: ['AED', 'Ventilator', 'Trauma Kit'],
  },
  {
    id: 'u-fire-1',
    callsign: 'Engine-5',
    agency: 'fire',
    status: 'available',
    location: { lat: 37.778, lon: -122.417 },
    homeBase: { lat: 37.776, lon: -122.416 },
    capabilities: ['pump', 'ladder', 'wildfire'],
    crewSize: 4,
    lastStatusChange: new Date(),
    equipment: ['Hose', 'Foam', 'Thermal Camera'],
  },
  {
    id: 'u-police-1',
    callsign: 'Unit-42',
    agency: 'police',
    status: 'available',
    location: { lat: 37.772, lon: -122.423 },
    homeBase: { lat: 37.771, lon: -122.424 },
    capabilities: ['traffic_control', 'investigation', 'k9'],
    crewSize: 2,
    lastStatusChange: new Date(),
  },
];

const defaultSLA: SLAProfile = {
  id: 'sla-default',
  name: 'Urban Multi-Agency SLA',
  agency: 'multi',
  coverageArea: { north: 90, south: -90, east: 180, west: -180 },
  targetResponseSeconds: 480,
  escalationThresholdSeconds: 600,
  priorityOverrides: [
    { priority: 1, targetResponseSeconds: 300 },
    { priority: 2, targetResponseSeconds: 420 },
  ],
};

// Utilities
function uid(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function distanceMeters(a: GeoPoint, b: GeoPoint): number {
  const R = 6371e3;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLon = toRad(b.lon - a.lon);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const sinDlat = Math.sin(dLat / 2);
  const sinDlon = Math.sin(dLon / 2);
  const h = sinDlat * sinDlat + Math.cos(lat1) * Math.cos(lat2) * sinDlon * sinDlon;
  return 2 * R * Math.asin(Math.sqrt(h));
}

function estimateDurationSeconds(distanceM: number, speedKph = 40): number {
  const speedMps = (speedKph * 1000) / 3600; // naive default urban speed
  return Math.round(distanceM / speedMps);
}

// Service implementation
class EmergencyDispatchService {
  private static instance: EmergencyDispatchService;

  private incidents: Map<string, Incident> = new Map();
  private units: Map<string, Unit> = new Map();
  private assignments: Map<string, Assignment> = new Map();
  private routes: Map<string, RoutePlan> = new Map();
  private slas: Map<string, SLAProfile> = new Map();
  private hooks: IntegrationHooks = {};

  private constructor() {
    // Initialize roster and default SLA
    sampleUnits.forEach((u) => this.units.set(u.id, { ...u }));
    this.slas.set(defaultSLA.id, defaultSLA);
  }

  public static getInstance(): EmergencyDispatchService {
    if (!EmergencyDispatchService.instance) {
      EmergencyDispatchService.instance = new EmergencyDispatchService();
    }
    return EmergencyDispatchService.instance;
  }

  // Integration hooks registration
  registerHooks(hooks: IntegrationHooks) {
    this.hooks = { ...this.hooks, ...hooks };
  }

  // Incident lifecycle
  async createIncident(input: Omit<Incident, 'id' | 'status' | 'createdAt' | 'updatedAt' | 'assignments' | 'history'>): Promise<Incident> {
    const id = uid('inc');
    const now = new Date();
    const incident: Incident = {
      ...input,
      id,
      status: 'created',
      createdAt: now,
      updatedAt: now,
      assignments: [],
      history: [
        {
          id: uid('evt'),
          type: 'created',
          message: `Incident ${id} created with priority ${input.priority}`,
          createdAt: now,
        },
      ],
    };
    this.incidents.set(id, incident);
    await this.emitNotify(incident, 'created', ['dispatcher'], `New ${incident.type} incident created`);
    await this.emitAudit('dispatch', 'incident_created', id, { priority: incident.priority, type: incident.type });
    return incident;
  }

  async updateIncident(incidentId: string, updates: Partial<Incident>): Promise<Incident> {
    const inc = this.mustGetIncident(incidentId);
    Object.assign(inc, updates);
    inc.updatedAt = new Date();
    this.appendHistory(inc, 'note', 'Incident updated', updates);
    await this.emitAudit('dispatch', 'incident_updated', inc.id, updates);
    return inc;
  }

  async setIncidentPriority(incidentId: string, priority: Priority): Promise<Incident> {
    const inc = this.mustGetIncident(incidentId);
    inc.priority = priority;
    inc.updatedAt = new Date();
    this.appendHistory(inc, 'priority_change', `Priority changed to ${priority}`);
    await this.emitAudit('dispatch', 'priority_change', inc.id, { priority });
    return inc;
  }

  async setIncidentStatus(incidentId: string, status: DispatchStatus): Promise<Incident> {
    const inc = this.mustGetIncident(incidentId);
    inc.status = status;
    if (status === 'cleared' || status === 'cancelled') {
      inc.clearedAt = new Date();
    }
    inc.updatedAt = new Date();
    this.appendHistory(inc, 'status_change', `Status set to ${status}`);
    await this.emitAudit('dispatch', 'status_change', inc.id, { status });
    await this.emitNotify(inc, 'status_change', ['dispatcher'], `Incident ${inc.id} status: ${status}`);
    return inc;
  }

  // Unit management
  async upsertUnit(unit: Unit): Promise<Unit> {
    const existing = this.units.get(unit.id);
    const merged: Unit = existing ? { ...existing, ...unit, lastStatusChange: new Date() } : { ...unit, lastStatusChange: new Date() };
    this.units.set(merged.id, merged);
    await this.emitAudit('dispatch', 'unit_upsert', merged.id, { status: merged.status });
    return merged;
  }

  async setUnitStatus(unitId: string, status: UnitStatus, location?: GeoPoint): Promise<Unit> {
    const unit = this.mustGetUnit(unitId);
    unit.status = status;
    if (location) unit.location = location;
    unit.lastStatusChange = new Date();
    await this.emitAudit('dispatch', 'unit_status', unit.id, { status });
    return unit;
  }

  // Assignment
  async assignUnits(incidentId: string, unitIds: string[], role: Assignment['role'] = 'primary'): Promise<Assignment[]> {
    const inc = this.mustGetIncident(incidentId);
    const created: Assignment[] = [];

    for (const uid_ of unitIds) {
      const unit = this.mustGetUnit(uid_);
      const route = await this.planRoute(unit.location, inc.location, { incidentId: inc.id });
      const assignment: Assignment = {
        id: uid('asg'),
        incidentId: inc.id,
        unitId: unit.id,
        assignedAt: new Date(),
        status: 'assigned',
        etaSeconds: route?.durationSeconds ?? estimateDurationSeconds(distanceMeters(unit.location, inc.location)),
        routeId: route?.id,
        role,
      };
      this.assignments.set(assignment.id, assignment);
      inc.assignments.push(assignment);
      unit.assignedIncidentId = inc.id;
      unit.status = 'enroute';
      unit.etaSeconds = assignment.etaSeconds;
      unit.lastStatusChange = new Date();
      this.appendHistory(inc, 'assignment_added', `Assigned ${unit.callsign} (${unit.agency})`, { unitId: unit.id });
      await this.emitAudit('assignment', 'unit_assigned', assignment.id, { incidentId: inc.id, unitId: unit.id });
      created.push(assignment);
    }

    inc.status = inc.status === 'created' ? 'queued' : inc.status;
    inc.updatedAt = new Date();
    await this.emitNotify(inc, 'assignment', ['dispatcher'], `${unitIds.length} unit(s) assigned to ${inc.id}`);
    return created;
  }

  async acknowledgeAssignment(assignmentId: string): Promise<Assignment> {
    const asg = this.mustGetAssignment(assignmentId);
    asg.status = 'acknowledged';
    asg.acknowledgedAt = new Date();
    const inc = this.mustGetIncident(asg.incidentId);
    this.appendHistory(inc, 'note', `Assignment ${asg.id} acknowledged`, { assignmentId: asg.id });
    await this.emitAudit('assignment', 'assignment_ack', asg.id, {});
    return asg;
  }

  async cancelAssignment(assignmentId: string, reason?: string): Promise<Assignment> {
    const asg = this.mustGetAssignment(assignmentId);
    asg.status = 'cancelled';
    const unit = this.mustGetUnit(asg.unitId);
    const inc = this.mustGetIncident(asg.incidentId);
    unit.assignedIncidentId = undefined;
    unit.status = 'available';
    unit.etaSeconds = undefined;
    unit.lastStatusChange = new Date();
    this.appendHistory(inc, 'assignment_removed', `Assignment ${asg.id} cancelled`, { reason });
    await this.emitAudit('assignment', 'assignment_cancel', asg.id, { reason });
    return asg;
  }

  // Auto-dispatch based on capability, distance, and priority
  async autoDispatch(
    incidentId: string,
    desired: { agency?: Agency; capabilities?: UnitCapability[]; count?: number } = { count: 1 }
  ): Promise<Assignment[]> {
    const inc = this.mustGetIncident(incidentId);
    const units = Array.from(this.units.values()).filter((u) => u.status === 'available');

    const filtered = units.filter((u) => {
      if (desired.agency && u.agency !== desired.agency) return false;
      if (desired.capabilities && desired.capabilities.length) {
        return desired.capabilities.every((c) => u.capabilities.includes(c));
      }
      return true;
    });

    const ranked = filtered
      .map((u) => ({
        unit: u,
        distance: distanceMeters(u.location, inc.location),
      }))
      .sort((a, b) => a.distance - b.distance);

    const count = desired.count ?? 1;
    const chosen = ranked.slice(0, count).map((r) => r.unit.id);
    return this.assignUnits(incidentId, chosen);
  }

  // Routing
  private async planRoute(origin: GeoPoint, destination: GeoPoint, context?: Record<string, unknown>): Promise<RoutePlan | undefined> {
    if (this.hooks.route) {
      try {
        const rp = await this.hooks.route(origin, destination, context);
        if (rp) {
          this.routes.set(rp.id, rp);
          return rp;
        }
      } catch {}
    }
    // Fallback naive route
    const dist = distanceMeters(origin, destination);
    const rp: RoutePlan = {
      id: uid('rte'),
      origin,
      destination,
      distanceMeters: dist,
      durationSeconds: estimateDurationSeconds(dist),
      createdAt: new Date(),
    };
    this.routes.set(rp.id, rp);
    return rp;
  }

  // Search and retrieval
  async getIncident(id: string): Promise<Incident | null> {
    return this.incidents.get(id) ?? null;
  }

  async listIncidents(filter?: {
    status?: DispatchStatus[];
    types?: IncidentType[];
    priority?: Priority[];
    since?: Date;
    until?: Date;
  }): Promise<Incident[]> {
    let items = Array.from(this.incidents.values());
    if (filter?.status) items = items.filter((i) => filter.status!.includes(i.status));
    if (filter?.types) items = items.filter((i) => filter.types!.includes(i.type));
    if (filter?.priority) items = items.filter((i) => filter.priority!.includes(i.priority));
    if (filter?.since) items = items.filter((i) => i.createdAt >= filter.since!);
    if (filter?.until) items = items.filter((i) => i.createdAt <= filter.until!);
    return items.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  async listUnits(filter?: { agency?: Agency; status?: UnitStatus[]; capabilitiesAll?: UnitCapability[] }): Promise<Unit[]> {
    let items = Array.from(this.units.values());
    if (filter?.agency) items = items.filter((u) => u.agency === filter.agency);
    if (filter?.status) items = items.filter((u) => filter.status!.includes(u.status));
    if (filter?.capabilitiesAll && filter.capabilitiesAll.length) {
      items = items.filter((u) => filter!.capabilitiesAll!.every((c) => u.capabilities.includes(c)));
    }
    return items;
  }

  // Coverage estimate: share of area within SLA
  async estimateCoverage(bounds: GeoBounds, gridSize = 5): Promise<{ coverageRate: number; samplePoints: number }>{
    const available = Array.from(this.units.values()).filter((u) => u.status === 'available');
    if (!available.length) return { coverageRate: 0, samplePoints: 0 };

    const latStep = (bounds.north - bounds.south) / gridSize;
    const lonStep = (bounds.east - bounds.west) / gridSize;
    const points: GeoPoint[] = [];
    for (let i = 0; i <= gridSize; i++) {
      for (let j = 0; j <= gridSize; j++) {
        points.push({ lat: bounds.south + latStep * i, lon: bounds.west + lonStep * j });
      }
    }

    const sla = defaultSLA; // naive
    let within = 0;
    for (const p of points) {
      const nearest = available
        .map((u) => estimateDurationSeconds(distanceMeters(u.location, p)))
        .sort((a, b) => a - b)[0];
      if (nearest <= sla.targetResponseSeconds) within++;
    }

    return { coverageRate: within / points.length, samplePoints: points.length };
  }

  // KPIs
  async getKPIs(): Promise<DispatchKPIs> {
    const incidents = Array.from(this.incidents.values());
    const active = incidents.filter((i) => i.status !== 'cleared' && i.status !== 'cancelled');

    const timeToAssign: number[] = [];
    const responseTimes: number[] = [];
    const clearTimes: number[] = [];
    const typeCounts: Record<string, number> = {};
    const priorityCounts: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };

    for (const inc of incidents) {
      typeCounts[inc.type] = (typeCounts[inc.type] ?? 0) + 1;
      priorityCounts[inc.priority] = (priorityCounts[inc.priority] ?? 0) + 1;

      if (inc.assignments.length) {
        const firstAssign = inc.assignments.map((a) => a.assignedAt.getTime()).sort((a, b) => a - b)[0];
        timeToAssign.push(firstAssign - inc.createdAt.getTime());
      }
      const firstEta = inc.assignments.map((a) => a.etaSeconds ?? 0).filter(Boolean).sort((a, b) => a - b)[0];
      if (firstEta) responseTimes.push(firstEta * 1000);
      if (inc.clearedAt) clearTimes.push(inc.clearedAt.getTime() - inc.createdAt.getTime());
    }

    const avg = (arr: number[]) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0);
    const utilizationByAgency = this.computeUtilization();
    const slaTargetMs = (defaultSLA.targetResponseSeconds || 480) * 1000;
    const onTime = responseTimes.filter((rt) => rt <= slaTargetMs).length;

    return {
      totalIncidents: incidents.length,
      activeIncidents: active.length,
      averageTimeToAssign: Math.round(avg(timeToAssign) / 1000),
      averageResponseTime: Math.round(avg(responseTimes) / 1000),
      onTimeRate: responseTimes.length ? onTime / responseTimes.length : 0,
      averageClearTime: Math.round(avg(clearTimes) / 1000),
      multiAgencyIncidents: incidents.filter((i) => new Set(i.assignments.map((a) => this.mustGetUnit(a.unitId).agency)).size > 1).length,
      unitUtilization: utilizationByAgency,
      incidentsByType: Object.entries(typeCounts).map(([type, count]) => ({ type: type as IncidentType, count })),
      priorityMix: (Object.keys(priorityCounts) as unknown as Priority[]).map((p: any) => ({ priority: Number(p) as Priority, count: (priorityCounts as any)[p] })),
    };
  }

  private computeUtilization(): { agency: Agency; utilization: number }[] {
    const units = Array.from(this.units.values());
    const byAgency = new Map<Agency, { total: number; busy: number }>();
    for (const u of units) {
      const agg = byAgency.get(u.agency) ?? { total: 0, busy: 0 };
      agg.total += 1;
      if (u.status !== 'available' && u.status !== 'maintenance' && u.status !== 'training') agg.busy += 1;
      byAgency.set(u.agency, agg);
    }
    return Array.from(byAgency.entries()).map(([agency, { total, busy }]) => ({ agency, utilization: total ? busy / total : 0 }));
  }

  // Helpers
  private mustGetIncident(id: string): Incident {
    const inc = this.incidents.get(id);
    if (!inc) throw new Error(`Incident not found: ${id}`);
    return inc;
    }

  private mustGetUnit(id: string): Unit {
    const u = this.units.get(id);
    if (!u) throw new Error(`Unit not found: ${id}`);
    return u;
  }

  private mustGetAssignment(id: string): Assignment {
    const a = this.assignments.get(id);
    if (!a) throw new Error(`Assignment not found: ${id}`);
    return a;
  }

  private appendHistory(inc: Incident, type: IncidentEvent['type'], message: string, data?: Record<string, unknown>) {
    inc.history.push({ id: uid('evt'), type, message, createdAt: new Date(), data });
  }

  private async emitNotify(incident: Incident, action: string, audience: ('unit' | 'supervisor' | 'dispatcher' | 'public')[], message: string) {
    if (this.hooks.notify) {
      try {
        await this.hooks.notify({ incident, action, audience, message });
      } catch {}
    }
  }

  private async emitAudit(category: 'dispatch' | 'assignment' | 'status' | 'kpi', action: string, entityId: string, metadata?: Record<string, unknown>) {
    if (this.hooks.audit) {
      try {
        await this.hooks.audit({ category, action, entityId, metadata, at: new Date() });
      } catch {}
    }
  }
}

export const emergencyDispatchService = EmergencyDispatchService.getInstance();
export default EmergencyDispatchService;
