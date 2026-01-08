/**
 * Audit Logging Service
 * Comprehensive audit trail, security logging, and compliance tracking
 */

// Log level
type LogLevel = 'debug' | 'info' | 'warning' | 'error' | 'critical';

// Event category
type EventCategory = 'authentication' | 'authorization' | 'data_access' | 'data_modification' | 'system' | 'security' | 'compliance' | 'user_activity' | 'api' | 'emergency';

// Event status
type EventStatus = 'success' | 'failure' | 'partial' | 'pending';

// Actor type
type ActorType = 'user' | 'system' | 'api' | 'service' | 'admin' | 'external';

// Resource type
type ResourceType = 'user' | 'alert' | 'shelter' | 'donation' | 'resource' | 'volunteer' | 'report' | 'file' | 'setting' | 'permission';

// Retention policy
type RetentionPolicy = '30_days' | '90_days' | '1_year' | '3_years' | '7_years' | 'indefinite';

// Audit event
interface AuditEvent {
  id: string;
  timestamp: Date;
  level: LogLevel;
  category: EventCategory;
  action: string;
  status: EventStatus;
  actor: {
    id: string;
    type: ActorType;
    name?: string;
    email?: string;
    ip?: string;
    userAgent?: string;
    sessionId?: string;
  };
  resource?: {
    type: ResourceType;
    id: string;
    name?: string;
  };
  details: Record<string, unknown>;
  changes?: {
    field: string;
    oldValue: unknown;
    newValue: unknown;
  }[];
  metadata: {
    correlationId?: string;
    requestId?: string;
    traceId?: string;
    spanId?: string;
    source: string;
    environment: string;
    version: string;
  };
  location?: {
    latitude: number;
    longitude: number;
    city?: string;
    country?: string;
  };
  risk?: {
    score: number;
    level: 'low' | 'medium' | 'high' | 'critical';
    indicators: string[];
  };
  tags: string[];
  hash?: string;
  signature?: string;
}

// Audit log filter
interface AuditLogFilter {
  startDate?: Date;
  endDate?: Date;
  levels?: LogLevel[];
  categories?: EventCategory[];
  actions?: string[];
  actorIds?: string[];
  actorTypes?: ActorType[];
  resourceTypes?: ResourceType[];
  resourceIds?: string[];
  status?: EventStatus[];
  search?: string;
  tags?: string[];
  minRiskScore?: number;
  correlationId?: string;
}

// Audit log summary
interface AuditLogSummary {
  totalEvents: number;
  byLevel: Record<LogLevel, number>;
  byCategory: Record<EventCategory, number>;
  byStatus: Record<EventStatus, number>;
  byHour: { hour: string; count: number }[];
  topActions: { action: string; count: number }[];
  topActors: { actorId: string; name: string; count: number }[];
  riskSummary: {
    totalHighRisk: number;
    totalCriticalRisk: number;
    topIndicators: { indicator: string; count: number }[];
  };
}

// Audit policy
interface AuditPolicy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  categories: EventCategory[];
  actions: string[];
  retentionPolicy: RetentionPolicy;
  alertOnFailure: boolean;
  alertOnHighRisk: boolean;
  requireSignature: boolean;
  includeDetails: boolean;
  includeChanges: boolean;
  excludeFields: string[];
  createdAt: Date;
  updatedAt: Date;
}

// Compliance report
interface ComplianceReport {
  id: string;
  name: string;
  type: 'SOC2' | 'GDPR' | 'HIPAA' | 'PCI_DSS' | 'ISO27001' | 'custom';
  period: { start: Date; end: Date };
  generatedAt: Date;
  status: 'generating' | 'completed' | 'failed';
  summary: {
    totalEvents: number;
    complianceScore: number;
    violations: number;
    warnings: number;
  };
  sections: {
    title: string;
    description: string;
    status: 'pass' | 'fail' | 'warning' | 'not_applicable';
    evidence: string[];
    recommendations: string[];
  }[];
  generatedBy: string;
}

// Alert rule
interface AuditAlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  conditions: {
    field: string;
    operator: 'equals' | 'contains' | 'greater_than' | 'less_than' | 'in' | 'not_in' | 'regex';
    value: unknown;
  }[];
  threshold?: {
    count: number;
    timeWindow: number; // minutes
  };
  actions: {
    type: 'email' | 'sms' | 'webhook' | 'slack' | 'pagerduty';
    target: string;
  }[];
  severity: LogLevel;
  cooldown: number; // minutes
  lastTriggered?: Date;
  triggerCount: number;
  createdAt: Date;
  updatedAt: Date;
}

// Export configuration
interface ExportConfig {
  format: 'json' | 'csv' | 'pdf' | 'xml';
  filter: AuditLogFilter;
  fields: string[];
  includeMetadata: boolean;
  compress: boolean;
  encrypt: boolean;
  destination?: {
    type: 'download' | 's3' | 'sftp' | 'email';
    config: Record<string, unknown>;
  };
}

// Archived log
interface ArchivedLog {
  id: string;
  period: { start: Date; end: Date };
  eventCount: number;
  fileSize: number;
  filePath: string;
  hash: string;
  encrypted: boolean;
  archivedAt: Date;
  expiresAt?: Date;
  metadata: Record<string, unknown>;
}

// Session activity
interface SessionActivity {
  sessionId: string;
  userId: string;
  startedAt: Date;
  lastActivityAt: Date;
  endedAt?: Date;
  ipAddress: string;
  userAgent: string;
  device?: {
    type: string;
    os: string;
    browser: string;
  };
  location?: {
    city: string;
    country: string;
  };
  events: number;
  riskScore: number;
  isActive: boolean;
}

// Security incident
interface SecurityIncident {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'investigating' | 'contained' | 'resolved' | 'closed';
  category: string;
  affectedResources: { type: ResourceType; id: string; name?: string }[];
  affectedUsers: string[];
  relatedEvents: string[];
  timeline: { timestamp: Date; action: string; actor: string; notes?: string }[];
  assignedTo?: string;
  rootCause?: string;
  resolution?: string;
  preventionMeasures?: string[];
  reportedAt: Date;
  resolvedAt?: Date;
  createdAt: Date;
  updatedAt: Date;
}

// Data access log
interface DataAccessLog {
  id: string;
  timestamp: Date;
  userId: string;
  resourceType: ResourceType;
  resourceId: string;
  accessType: 'view' | 'download' | 'export' | 'print' | 'share';
  fields?: string[];
  purpose?: string;
  ipAddress: string;
  sessionId: string;
  authorized: boolean;
  metadata: Record<string, unknown>;
}

// API access log
interface APIAccessLog {
  id: string;
  timestamp: Date;
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  endpoint: string;
  statusCode: number;
  responseTime: number; // ms
  requestSize: number; // bytes
  responseSize: number; // bytes
  clientId?: string;
  userId?: string;
  ipAddress: string;
  userAgent: string;
  error?: string;
  rateLimit?: {
    remaining: number;
    limit: number;
    reset: Date;
  };
}

class AuditLoggingService {
  private static instance: AuditLoggingService;
  private events: Map<string, AuditEvent> = new Map();
  private policies: Map<string, AuditPolicy> = new Map();
  private alertRules: Map<string, AuditAlertRule> = new Map();
  private complianceReports: Map<string, ComplianceReport> = new Map();
  private archivedLogs: ArchivedLog[] = [];
  private sessions: Map<string, SessionActivity> = new Map();
  private incidents: Map<string, SecurityIncident> = new Map();
  private dataAccessLogs: DataAccessLog[] = [];
  private apiAccessLogs: APIAccessLog[] = [];
  private listeners: ((event: string, data: unknown) => void)[] = [];
  private environment: string = 'production';
  private version: string = '1.0.0';

  private constructor() {
    this.initializeSampleData();
  }

  public static getInstance(): AuditLoggingService {
    if (!AuditLoggingService.instance) {
      AuditLoggingService.instance = new AuditLoggingService();
    }
    return AuditLoggingService.instance;
  }

  /**
   * Initialize sample data
   */
  private initializeSampleData(): void {
    // Initialize audit policies
    const policiesData = [
      { id: 'policy-auth', name: 'Authentication Events', description: 'Log all authentication events', categories: ['authentication' as EventCategory], actions: ['login', 'logout', 'password_change', 'mfa_setup'], retentionPolicy: '1_year' as RetentionPolicy, alertOnFailure: true },
      { id: 'policy-security', name: 'Security Events', description: 'Log all security-related events', categories: ['security' as EventCategory, 'authorization' as EventCategory], actions: ['permission_change', 'role_assignment', 'access_denied'], retentionPolicy: '3_years' as RetentionPolicy, alertOnFailure: true, alertOnHighRisk: true },
      { id: 'policy-data', name: 'Data Access Events', description: 'Log data access and modifications', categories: ['data_access' as EventCategory, 'data_modification' as EventCategory], actions: ['view', 'create', 'update', 'delete', 'export'], retentionPolicy: '1_year' as RetentionPolicy, alertOnFailure: false },
      { id: 'policy-emergency', name: 'Emergency Events', description: 'Log all emergency-related actions', categories: ['emergency' as EventCategory], actions: ['alert_created', 'alert_acknowledged', 'evacuation_ordered', 'sos_triggered'], retentionPolicy: '7_years' as RetentionPolicy, alertOnFailure: true },
      { id: 'policy-compliance', name: 'Compliance Events', description: 'Log compliance-related activities', categories: ['compliance' as EventCategory], actions: ['audit_access', 'report_generated', 'policy_updated'], retentionPolicy: '7_years' as RetentionPolicy, alertOnFailure: false, requireSignature: true },
    ];

    policiesData.forEach((p) => {
      const policy: AuditPolicy = {
        ...p,
        enabled: true,
        alertOnHighRisk: p.alertOnHighRisk || false,
        requireSignature: p.requireSignature || false,
        includeDetails: true,
        includeChanges: true,
        excludeFields: ['password', 'token', 'secret'],
        createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
        updatedAt: new Date(),
      };
      this.policies.set(policy.id, policy);
    });

    // Initialize alert rules
    const alertRulesData = [
      { id: 'rule-failed-login', name: 'Multiple Failed Logins', description: 'Alert on multiple failed login attempts', conditions: [{ field: 'action', operator: 'equals' as const, value: 'login' }, { field: 'status', operator: 'equals' as const, value: 'failure' }], threshold: { count: 5, timeWindow: 15 }, severity: 'warning' as LogLevel },
      { id: 'rule-data-export', name: 'Large Data Export', description: 'Alert on large data exports', conditions: [{ field: 'action', operator: 'equals' as const, value: 'export' }, { field: 'details.recordCount', operator: 'greater_than' as const, value: 1000 }], severity: 'info' as LogLevel },
      { id: 'rule-admin-action', name: 'Admin Action', description: 'Alert on all admin actions', conditions: [{ field: 'actor.type', operator: 'equals' as const, value: 'admin' }], severity: 'info' as LogLevel },
      { id: 'rule-high-risk', name: 'High Risk Activity', description: 'Alert on high risk activities', conditions: [{ field: 'risk.level', operator: 'in' as const, value: ['high', 'critical'] }], severity: 'critical' as LogLevel },
    ];

    alertRulesData.forEach((r) => {
      const rule: AuditAlertRule = {
        ...r,
        enabled: true,
        actions: [{ type: 'email', target: 'security@alertaid.com' }],
        cooldown: 60,
        triggerCount: 0,
        createdAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000),
        updatedAt: new Date(),
      };
      this.alertRules.set(rule.id, rule);
    });

    // Initialize sample audit events
    const actions = [
      { action: 'login', category: 'authentication' as EventCategory, level: 'info' as LogLevel },
      { action: 'logout', category: 'authentication' as EventCategory, level: 'info' as LogLevel },
      { action: 'password_change', category: 'authentication' as EventCategory, level: 'info' as LogLevel },
      { action: 'alert_created', category: 'emergency' as EventCategory, level: 'warning' as LogLevel },
      { action: 'shelter_updated', category: 'data_modification' as EventCategory, level: 'info' as LogLevel },
      { action: 'donation_received', category: 'data_modification' as EventCategory, level: 'info' as LogLevel },
      { action: 'user_created', category: 'data_modification' as EventCategory, level: 'info' as LogLevel },
      { action: 'permission_granted', category: 'authorization' as EventCategory, level: 'info' as LogLevel },
      { action: 'api_access', category: 'api' as EventCategory, level: 'debug' as LogLevel },
      { action: 'report_viewed', category: 'data_access' as EventCategory, level: 'info' as LogLevel },
    ];

    const statuses: EventStatus[] = ['success', 'success', 'success', 'success', 'failure'];
    const actorTypes: ActorType[] = ['user', 'user', 'user', 'admin', 'system', 'api'];

    for (let i = 0; i < 100; i++) {
      const actionData = actions[i % actions.length];
      const timestamp = new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000);
      const status = statuses[Math.floor(Math.random() * statuses.length)];

      const event: AuditEvent = {
        id: `event-${(i + 1).toString().padStart(8, '0')}`,
        timestamp,
        level: status === 'failure' ? 'error' : actionData.level,
        category: actionData.category,
        action: actionData.action,
        status,
        actor: {
          id: `user-${(Math.floor(Math.random() * 50) + 1)}`,
          type: actorTypes[Math.floor(Math.random() * actorTypes.length)],
          name: `User ${Math.floor(Math.random() * 50) + 1}`,
          ip: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
          sessionId: `session-${Math.random().toString(36).substr(2, 10)}`,
        },
        details: {
          message: `${actionData.action} performed`,
          timestamp: timestamp.toISOString(),
        },
        metadata: {
          source: 'alert-aid-web',
          environment: this.environment,
          version: this.version,
          correlationId: `corr-${Math.random().toString(36).substr(2, 10)}`,
        },
        tags: [actionData.category, status],
        risk: status === 'failure' ? {
          score: Math.floor(Math.random() * 50) + 50,
          level: Math.random() > 0.7 ? 'high' : 'medium',
          indicators: ['failed_attempt', 'unusual_time'],
        } : undefined,
      };

      this.events.set(event.id, event);
    }

    // Initialize sample sessions
    for (let i = 1; i <= 30; i++) {
      const session: SessionActivity = {
        sessionId: `session-${i.toString().padStart(6, '0')}`,
        userId: `user-${(i % 20) + 1}`,
        startedAt: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
        lastActivityAt: new Date(Date.now() - Math.random() * 60 * 60 * 1000),
        ipAddress: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        device: {
          type: Math.random() > 0.5 ? 'desktop' : 'mobile',
          os: Math.random() > 0.5 ? 'Windows' : 'Android',
          browser: 'Chrome',
        },
        location: {
          city: ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'][Math.floor(Math.random() * 5)],
          country: 'India',
        },
        events: Math.floor(Math.random() * 50) + 5,
        riskScore: Math.floor(Math.random() * 30),
        isActive: Math.random() > 0.3,
      };
      this.sessions.set(session.sessionId, session);
    }

    // Initialize sample incidents
    const incidentsData = [
      { title: 'Brute Force Attack Detected', description: 'Multiple failed login attempts from single IP', severity: 'high' as const, category: 'authentication', status: 'investigating' as const },
      { title: 'Unauthorized Data Access', description: 'User attempted to access restricted shelter data', severity: 'medium' as const, category: 'authorization', status: 'resolved' as const },
      { title: 'Suspicious API Activity', description: 'Unusual API call patterns detected', severity: 'low' as const, category: 'api', status: 'closed' as const },
    ];

    incidentsData.forEach((inc, index) => {
      const incident: SecurityIncident = {
        id: `incident-${(index + 1).toString().padStart(6, '0')}`,
        ...inc,
        affectedResources: [],
        affectedUsers: [`user-${index + 1}`],
        relatedEvents: [`event-${(index + 1).toString().padStart(8, '0')}`],
        timeline: [
          { timestamp: new Date(Date.now() - 3600000), action: 'detected', actor: 'system', notes: 'Automated detection' },
          { timestamp: new Date(Date.now() - 1800000), action: 'assigned', actor: 'admin-1', notes: 'Assigned for investigation' },
        ],
        reportedAt: new Date(Date.now() - 3600000),
        createdAt: new Date(Date.now() - 3600000),
        updatedAt: new Date(),
      };
      this.incidents.set(incident.id, incident);
    });

    // Initialize API access logs
    const endpoints = ['/api/alerts', '/api/shelters', '/api/users', '/api/donations', '/api/resources'];
    const methods: ('GET' | 'POST' | 'PUT' | 'DELETE')[] = ['GET', 'GET', 'GET', 'POST', 'PUT', 'DELETE'];

    for (let i = 0; i < 100; i++) {
      const log: APIAccessLog = {
        id: `api-${(i + 1).toString().padStart(8, '0')}`,
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
        method: methods[Math.floor(Math.random() * methods.length)],
        endpoint: endpoints[Math.floor(Math.random() * endpoints.length)],
        statusCode: Math.random() > 0.9 ? [400, 401, 403, 404, 500][Math.floor(Math.random() * 5)] : 200,
        responseTime: Math.floor(Math.random() * 500) + 50,
        requestSize: Math.floor(Math.random() * 5000),
        responseSize: Math.floor(Math.random() * 50000),
        userId: `user-${Math.floor(Math.random() * 50) + 1}`,
        ipAddress: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
        userAgent: 'AlertAid/1.0',
        rateLimit: {
          remaining: Math.floor(Math.random() * 1000),
          limit: 1000,
          reset: new Date(Date.now() + 60000),
        },
      };
      this.apiAccessLogs.push(log);
    }
  }

  /**
   * Log event
   */
  public log(data: {
    level: LogLevel;
    category: EventCategory;
    action: string;
    status: EventStatus;
    actor: AuditEvent['actor'];
    resource?: AuditEvent['resource'];
    details?: Record<string, unknown>;
    changes?: AuditEvent['changes'];
    location?: AuditEvent['location'];
    tags?: string[];
  }): AuditEvent {
    const event: AuditEvent = {
      id: `event-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`,
      timestamp: new Date(),
      ...data,
      details: data.details || {},
      tags: data.tags || [],
      metadata: {
        source: 'alert-aid',
        environment: this.environment,
        version: this.version,
        correlationId: `corr-${Math.random().toString(36).substr(2, 10)}`,
      },
    };

    // Calculate risk
    if (event.status === 'failure' || event.level === 'error' || event.level === 'critical') {
      event.risk = this.calculateRisk(event);
    }

    // Generate hash for integrity
    event.hash = this.generateHash(event);

    this.events.set(event.id, event);
    this.emit('event_logged', event);

    // Check alert rules
    this.checkAlertRules(event);

    return event;
  }

  /**
   * Calculate risk score
   */
  private calculateRisk(event: AuditEvent): AuditEvent['risk'] {
    let score = 0;
    const indicators: string[] = [];

    // Factor: Status
    if (event.status === 'failure') {
      score += 30;
      indicators.push('failed_action');
    }

    // Factor: Level
    if (event.level === 'error') {
      score += 20;
      indicators.push('error_level');
    } else if (event.level === 'critical') {
      score += 40;
      indicators.push('critical_level');
    }

    // Factor: Category
    if (event.category === 'security' || event.category === 'authentication') {
      score += 15;
      indicators.push('security_related');
    }

    // Factor: Time (unusual hours)
    const hour = event.timestamp.getHours();
    if (hour < 6 || hour > 22) {
      score += 10;
      indicators.push('unusual_time');
    }

    let level: 'low' | 'medium' | 'high' | 'critical';
    if (score < 25) level = 'low';
    else if (score < 50) level = 'medium';
    else if (score < 75) level = 'high';
    else level = 'critical';

    return { score: Math.min(score, 100), level, indicators };
  }

  /**
   * Generate hash
   */
  private generateHash(event: AuditEvent): string {
    const content = JSON.stringify({
      timestamp: event.timestamp.toISOString(),
      action: event.action,
      actor: event.actor.id,
      resource: event.resource?.id,
    });
    // Simple hash for demo
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(16, '0');
  }

  /**
   * Check alert rules
   */
  private checkAlertRules(event: AuditEvent): void {
    for (const rule of this.alertRules.values()) {
      if (!rule.enabled) continue;

      const matches = rule.conditions.every((condition) => {
        const value = this.getNestedValue(event, condition.field);
        switch (condition.operator) {
          case 'equals': return value === condition.value;
          case 'contains': return String(value).includes(String(condition.value));
          case 'greater_than': return Number(value) > Number(condition.value);
          case 'less_than': return Number(value) < Number(condition.value);
          case 'in': return Array.isArray(condition.value) && condition.value.includes(value);
          case 'not_in': return Array.isArray(condition.value) && !condition.value.includes(value);
          default: return false;
        }
      });

      if (matches) {
        // Check cooldown
        if (rule.lastTriggered) {
          const cooldownMs = rule.cooldown * 60 * 1000;
          if (Date.now() - rule.lastTriggered.getTime() < cooldownMs) continue;
        }

        rule.lastTriggered = new Date();
        rule.triggerCount++;
        this.emit('alert_triggered', { rule, event });
      }
    }
  }

  /**
   * Get nested value
   */
  private getNestedValue(obj: Record<string, unknown>, path: string): unknown {
    return path.split('.').reduce((current: unknown, key) => {
      return current && typeof current === 'object' ? (current as Record<string, unknown>)[key] : undefined;
    }, obj);
  }

  /**
   * Query events
   */
  public query(filter: AuditLogFilter, pagination?: { page: number; limit: number }): {
    events: AuditEvent[];
    total: number;
    page: number;
    limit: number;
    totalPages: number;
  } {
    let events = Array.from(this.events.values());

    // Apply filters
    if (filter.startDate) {
      events = events.filter((e) => e.timestamp >= filter.startDate!);
    }
    if (filter.endDate) {
      events = events.filter((e) => e.timestamp <= filter.endDate!);
    }
    if (filter.levels?.length) {
      events = events.filter((e) => filter.levels!.includes(e.level));
    }
    if (filter.categories?.length) {
      events = events.filter((e) => filter.categories!.includes(e.category));
    }
    if (filter.actions?.length) {
      events = events.filter((e) => filter.actions!.includes(e.action));
    }
    if (filter.actorIds?.length) {
      events = events.filter((e) => filter.actorIds!.includes(e.actor.id));
    }
    if (filter.actorTypes?.length) {
      events = events.filter((e) => filter.actorTypes!.includes(e.actor.type));
    }
    if (filter.resourceTypes?.length) {
      events = events.filter((e) => e.resource && filter.resourceTypes!.includes(e.resource.type));
    }
    if (filter.status?.length) {
      events = events.filter((e) => filter.status!.includes(e.status));
    }
    if (filter.search) {
      const search = filter.search.toLowerCase();
      events = events.filter((e) =>
        e.action.toLowerCase().includes(search) ||
        e.actor.name?.toLowerCase().includes(search) ||
        JSON.stringify(e.details).toLowerCase().includes(search)
      );
    }
    if (filter.minRiskScore !== undefined) {
      events = events.filter((e) => e.risk && e.risk.score >= filter.minRiskScore!);
    }
    if (filter.correlationId) {
      events = events.filter((e) => e.metadata.correlationId === filter.correlationId);
    }

    // Sort by timestamp desc
    events.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    const total = events.length;
    const page = pagination?.page || 1;
    const limit = pagination?.limit || 50;
    const totalPages = Math.ceil(total / limit);

    // Apply pagination
    const start = (page - 1) * limit;
    events = events.slice(start, start + limit);

    return { events, total, page, limit, totalPages };
  }

  /**
   * Get summary
   */
  public getSummary(filter?: AuditLogFilter): AuditLogSummary {
    const result = this.query(filter || {}, { page: 1, limit: 10000 });
    const events = result.events;

    const byLevel: Record<LogLevel, number> = { debug: 0, info: 0, warning: 0, error: 0, critical: 0 };
    const byCategory: Record<EventCategory, number> = { authentication: 0, authorization: 0, data_access: 0, data_modification: 0, system: 0, security: 0, compliance: 0, user_activity: 0, api: 0, emergency: 0 };
    const byStatus: Record<EventStatus, number> = { success: 0, failure: 0, partial: 0, pending: 0 };
    const actionCounts: Record<string, number> = {};
    const actorCounts: Record<string, { name: string; count: number }> = {};
    const hourCounts: Record<string, number> = {};
    let totalHighRisk = 0;
    let totalCriticalRisk = 0;
    const indicatorCounts: Record<string, number> = {};

    events.forEach((e) => {
      byLevel[e.level]++;
      byCategory[e.category]++;
      byStatus[e.status]++;

      actionCounts[e.action] = (actionCounts[e.action] || 0) + 1;

      if (!actorCounts[e.actor.id]) {
        actorCounts[e.actor.id] = { name: e.actor.name || e.actor.id, count: 0 };
      }
      actorCounts[e.actor.id].count++;

      const hour = e.timestamp.toISOString().slice(0, 13) + ':00';
      hourCounts[hour] = (hourCounts[hour] || 0) + 1;

      if (e.risk) {
        if (e.risk.level === 'high') totalHighRisk++;
        if (e.risk.level === 'critical') totalCriticalRisk++;
        e.risk.indicators.forEach((ind) => {
          indicatorCounts[ind] = (indicatorCounts[ind] || 0) + 1;
        });
      }
    });

    return {
      totalEvents: events.length,
      byLevel,
      byCategory,
      byStatus,
      byHour: Object.entries(hourCounts)
        .map(([hour, count]) => ({ hour, count }))
        .sort((a, b) => a.hour.localeCompare(b.hour)),
      topActions: Object.entries(actionCounts)
        .map(([action, count]) => ({ action, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10),
      topActors: Object.entries(actorCounts)
        .map(([actorId, data]) => ({ actorId, ...data }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10),
      riskSummary: {
        totalHighRisk,
        totalCriticalRisk,
        topIndicators: Object.entries(indicatorCounts)
          .map(([indicator, count]) => ({ indicator, count }))
          .sort((a, b) => b.count - a.count)
          .slice(0, 5),
      },
    };
  }

  /**
   * Get policies
   */
  public getPolicies(): AuditPolicy[] {
    return Array.from(this.policies.values());
  }

  /**
   * Get alert rules
   */
  public getAlertRules(): AuditAlertRule[] {
    return Array.from(this.alertRules.values());
  }

  /**
   * Get sessions
   */
  public getSessions(activeOnly: boolean = false): SessionActivity[] {
    let sessions = Array.from(this.sessions.values());
    if (activeOnly) {
      sessions = sessions.filter((s) => s.isActive);
    }
    return sessions.sort((a, b) => b.lastActivityAt.getTime() - a.lastActivityAt.getTime());
  }

  /**
   * Get incidents
   */
  public getIncidents(status?: SecurityIncident['status']): SecurityIncident[] {
    let incidents = Array.from(this.incidents.values());
    if (status) {
      incidents = incidents.filter((i) => i.status === status);
    }
    return incidents.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  /**
   * Get API access logs
   */
  public getAPIAccessLogs(filter?: { endpoint?: string; method?: string; startDate?: Date; endDate?: Date }): APIAccessLog[] {
    let logs = [...this.apiAccessLogs];

    if (filter?.endpoint) {
      logs = logs.filter((l) => l.endpoint.includes(filter.endpoint!));
    }
    if (filter?.method) {
      logs = logs.filter((l) => l.method === filter.method);
    }
    if (filter?.startDate) {
      logs = logs.filter((l) => l.timestamp >= filter.startDate!);
    }
    if (filter?.endDate) {
      logs = logs.filter((l) => l.timestamp <= filter.endDate!);
    }

    return logs.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  /**
   * Generate compliance report
   */
  public generateComplianceReport(type: ComplianceReport['type'], period: { start: Date; end: Date }, generatedBy: string): ComplianceReport {
    const filter: AuditLogFilter = { startDate: period.start, endDate: period.end };
    const summary = this.getSummary(filter);

    const report: ComplianceReport = {
      id: `report-${Date.now()}`,
      name: `${type} Compliance Report`,
      type,
      period,
      generatedAt: new Date(),
      status: 'completed',
      summary: {
        totalEvents: summary.totalEvents,
        complianceScore: Math.max(0, 100 - summary.riskSummary.totalCriticalRisk * 10 - summary.riskSummary.totalHighRisk * 5),
        violations: summary.riskSummary.totalCriticalRisk,
        warnings: summary.riskSummary.totalHighRisk,
      },
      sections: [
        {
          title: 'Access Control',
          description: 'Review of authentication and authorization controls',
          status: summary.byCategory.authorization > 0 && summary.byStatus.failure < 10 ? 'pass' : 'warning',
          evidence: ['Authentication logs', 'Authorization logs'],
          recommendations: [],
        },
        {
          title: 'Data Protection',
          description: 'Review of data access and modification controls',
          status: 'pass',
          evidence: ['Data access logs', 'Encryption status'],
          recommendations: [],
        },
        {
          title: 'Incident Response',
          description: 'Review of security incident handling',
          status: this.incidents.size > 0 ? 'warning' : 'pass',
          evidence: ['Incident logs', 'Response times'],
          recommendations: this.incidents.size > 0 ? ['Review incident handling procedures'] : [],
        },
      ],
      generatedBy,
    };

    this.complianceReports.set(report.id, report);
    return report;
  }

  /**
   * Subscribe to events
   */
  public subscribe(callback: (event: string, data: unknown) => void): () => void {
    this.listeners.push(callback);
    return () => {
      const index = this.listeners.indexOf(callback);
      if (index > -1) this.listeners.splice(index, 1);
    };
  }

  /**
   * Emit event
   */
  private emit(event: string, data: unknown): void {
    this.listeners.forEach((callback) => callback(event, data));
  }
}

export const auditLoggingService = AuditLoggingService.getInstance();
export type {
  LogLevel,
  EventCategory,
  EventStatus,
  ActorType,
  ResourceType,
  RetentionPolicy,
  AuditEvent,
  AuditLogFilter,
  AuditLogSummary,
  AuditPolicy,
  ComplianceReport,
  AuditAlertRule,
  ExportConfig,
  ArchivedLog,
  SessionActivity,
  SecurityIncident,
  DataAccessLog,
  APIAccessLog,
};
