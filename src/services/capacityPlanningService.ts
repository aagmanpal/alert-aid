/**
 * Capacity Planning Service
 * Comprehensive capacity management, forecasting, and resource optimization
 */

// Resource Type
type ResourceType = 'compute' | 'memory' | 'storage' | 'network' | 'database' | 'cache' | 'queue' | 'custom';

// Capacity Status
type CapacityStatus = 'healthy' | 'warning' | 'critical' | 'exhausted' | 'unknown';

// Trend Direction
type TrendDirection = 'increasing' | 'stable' | 'decreasing' | 'volatile';

// Forecast Confidence
type ForecastConfidence = 'high' | 'medium' | 'low' | 'very_low';

// Scaling Action
type ScalingAction = 'scale_up' | 'scale_down' | 'scale_out' | 'scale_in' | 'optimize' | 'migrate' | 'none';

// Resource
interface Resource {
  id: string;
  name: string;
  type: ResourceType;
  description: string;
  provider: ResourceProvider;
  configuration: ResourceConfiguration;
  capacity: ResourceCapacity;
  utilization: ResourceUtilization;
  thresholds: ResourceThresholds;
  forecast: ResourceForecast;
  recommendations: ResourceRecommendation[];
  tags: string[];
  labels: Record<string, string>;
  metadata: ResourceMetadata;
}

// Resource Provider
interface ResourceProvider {
  type: 'aws' | 'azure' | 'gcp' | 'on-premise' | 'hybrid' | 'kubernetes';
  region: string;
  zone?: string;
  account?: string;
  cluster?: string;
  namespace?: string;
}

// Resource Configuration
interface ResourceConfiguration {
  instanceType?: string;
  vcpus?: number;
  memoryGB?: number;
  storageGB?: number;
  iops?: number;
  throughputMBps?: number;
  networkBandwidthGbps?: number;
  connections?: number;
  replicas?: number;
  shards?: number;
  partitions?: number;
  customConfig?: Record<string, unknown>;
}

// Resource Capacity
interface ResourceCapacity {
  total: number;
  available: number;
  reserved: number;
  used: number;
  unit: string;
  scalable: boolean;
  minCapacity: number;
  maxCapacity: number;
  currentCost: number;
  costPerUnit: number;
  currency: string;
}

// Resource Utilization
interface ResourceUtilization {
  current: number;
  average: number;
  peak: number;
  percentile95: number;
  percentile99: number;
  trend: TrendDirection;
  trendPercentage: number;
  history: UtilizationDataPoint[];
  lastUpdated: Date;
}

// Utilization Data Point
interface UtilizationDataPoint {
  timestamp: Date;
  value: number;
  anomaly: boolean;
}

// Resource Thresholds
interface ResourceThresholds {
  warning: number;
  critical: number;
  targetUtilization: number;
  minUtilization: number;
  maxUtilization: number;
  scaleUpThreshold: number;
  scaleDownThreshold: number;
  cooldownPeriod: number;
}

// Resource Forecast
interface ResourceForecast {
  horizonDays: number;
  predictions: ForecastPrediction[];
  exhaustionDate?: Date;
  confidence: ForecastConfidence;
  model: string;
  accuracy: number;
  lastUpdated: Date;
}

// Forecast Prediction
interface ForecastPrediction {
  date: Date;
  predicted: number;
  lowerBound: number;
  upperBound: number;
  confidence: number;
}

// Resource Recommendation
interface ResourceRecommendation {
  id: string;
  type: 'scale' | 'optimize' | 'migrate' | 'rightsizing' | 'reservation' | 'spot';
  action: ScalingAction;
  priority: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  impact: RecommendationImpact;
  implementation: RecommendationImplementation;
  status: 'pending' | 'approved' | 'implemented' | 'rejected' | 'expired';
  createdAt: Date;
  expiresAt?: Date;
}

// Recommendation Impact
interface RecommendationImpact {
  capacityChange: number;
  costChange: number;
  costChangePercent: number;
  performanceImpact: 'positive' | 'neutral' | 'negative';
  riskLevel: 'low' | 'medium' | 'high';
  savingsPerMonth?: number;
}

// Recommendation Implementation
interface RecommendationImplementation {
  automated: boolean;
  steps: string[];
  estimatedDuration: number;
  downtime: boolean;
  approvalRequired: boolean;
  rollbackAvailable: boolean;
}

// Resource Metadata
interface ResourceMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  lastCapacityChange?: Date;
  version: number;
}

// Capacity Plan
interface CapacityPlan {
  id: string;
  name: string;
  description: string;
  scope: PlanScope;
  timeframe: PlanTimeframe;
  objectives: PlanObjective[];
  resources: PlanResource[];
  scenarios: CapacityScenario[];
  budgets: PlanBudget[];
  risks: PlanRisk[];
  milestones: PlanMilestone[];
  status: 'draft' | 'active' | 'approved' | 'completed' | 'archived';
  approvals: PlanApproval[];
  metadata: {
    createdAt: Date;
    createdBy: string;
    updatedAt: Date;
    version: number;
  };
}

// Plan Scope
interface PlanScope {
  services: string[];
  environments: string[];
  regions: string[];
  resourceTypes: ResourceType[];
  teams: string[];
}

// Plan Timeframe
interface PlanTimeframe {
  startDate: Date;
  endDate: Date;
  reviewCycle: 'weekly' | 'monthly' | 'quarterly';
  nextReview: Date;
}

// Plan Objective
interface PlanObjective {
  id: string;
  name: string;
  description: string;
  type: 'performance' | 'cost' | 'reliability' | 'growth' | 'compliance';
  target: number;
  current: number;
  unit: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  status: 'on_track' | 'at_risk' | 'behind' | 'achieved';
}

// Plan Resource
interface PlanResource {
  resourceId: string;
  resourceName: string;
  resourceType: ResourceType;
  currentCapacity: number;
  plannedCapacity: number;
  requiredCapacity: number;
  timeline: CapacityTimeline[];
  actions: PlannedAction[];
}

// Capacity Timeline
interface CapacityTimeline {
  date: Date;
  plannedCapacity: number;
  forecastedDemand: number;
  buffer: number;
}

// Planned Action
interface PlannedAction {
  id: string;
  action: ScalingAction;
  scheduledDate: Date;
  capacityChange: number;
  cost: number;
  status: 'planned' | 'scheduled' | 'in_progress' | 'completed' | 'cancelled';
  approvedBy?: string;
  approvedAt?: Date;
}

// Capacity Scenario
interface CapacityScenario {
  id: string;
  name: string;
  description: string;
  type: 'baseline' | 'optimistic' | 'pessimistic' | 'growth' | 'custom';
  assumptions: ScenarioAssumption[];
  projections: ScenarioProjection[];
  budget: number;
  likelihood: number;
}

// Scenario Assumption
interface ScenarioAssumption {
  name: string;
  value: number;
  unit: string;
  description: string;
}

// Scenario Projection
interface ScenarioProjection {
  month: number;
  demand: number;
  capacity: number;
  cost: number;
  utilizationPercent: number;
}

// Plan Budget
interface PlanBudget {
  id: string;
  category: 'compute' | 'storage' | 'network' | 'licensing' | 'support' | 'other';
  allocated: number;
  spent: number;
  forecast: number;
  variance: number;
  currency: string;
}

// Plan Risk
interface PlanRisk {
  id: string;
  name: string;
  description: string;
  category: 'capacity' | 'budget' | 'technical' | 'vendor' | 'compliance';
  likelihood: 'high' | 'medium' | 'low';
  impact: 'high' | 'medium' | 'low';
  mitigations: string[];
  owner: string;
  status: 'identified' | 'mitigated' | 'accepted' | 'occurred';
}

// Plan Milestone
interface PlanMilestone {
  id: string;
  name: string;
  description: string;
  targetDate: Date;
  actualDate?: Date;
  status: 'pending' | 'in_progress' | 'completed' | 'delayed' | 'cancelled';
  deliverables: string[];
}

// Plan Approval
interface PlanApproval {
  approver: string;
  role: string;
  status: 'pending' | 'approved' | 'rejected';
  date?: Date;
  comments?: string;
}

// Capacity Alert
interface CapacityAlert {
  id: string;
  resourceId: string;
  resourceName: string;
  resourceType: ResourceType;
  alertType: 'threshold' | 'forecast' | 'anomaly' | 'exhaustion' | 'cost';
  severity: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  currentValue: number;
  thresholdValue: number;
  triggeredAt: Date;
  acknowledgedAt?: Date;
  acknowledgedBy?: string;
  resolvedAt?: Date;
  resolvedBy?: string;
  status: 'active' | 'acknowledged' | 'resolved' | 'suppressed';
  actions: AlertAction[];
}

// Alert Action
interface AlertAction {
  type: 'scale' | 'notify' | 'ticket' | 'runbook';
  target: string;
  status: 'pending' | 'completed' | 'failed';
  executedAt?: Date;
  result?: string;
}

// Scaling Policy
interface ScalingPolicy {
  id: string;
  name: string;
  description: string;
  resourceType: ResourceType;
  resources: string[];
  triggers: ScalingTrigger[];
  actions: ScalingPolicyAction[];
  cooldown: number;
  schedule?: ScalingSchedule;
  limits: ScalingLimits;
  status: 'active' | 'disabled' | 'testing';
  metadata: {
    createdAt: Date;
    createdBy: string;
    updatedAt: Date;
    lastTriggered?: Date;
    triggerCount: number;
  };
}

// Scaling Trigger
interface ScalingTrigger {
  id: string;
  metric: string;
  condition: 'above' | 'below' | 'equals';
  threshold: number;
  duration: number;
  evaluationPeriods: number;
}

// Scaling Policy Action
interface ScalingPolicyAction {
  type: ScalingAction;
  adjustment: number;
  adjustmentType: 'absolute' | 'percent' | 'exact';
  minCapacity?: number;
  maxCapacity?: number;
}

// Scaling Schedule
interface ScalingSchedule {
  enabled: boolean;
  timezone: string;
  schedules: ScheduledScaling[];
}

// Scheduled Scaling
interface ScheduledScaling {
  id: string;
  name: string;
  cron: string;
  targetCapacity: number;
  enabled: boolean;
}

// Scaling Limits
interface ScalingLimits {
  minCapacity: number;
  maxCapacity: number;
  maxScaleUpStep: number;
  maxScaleDownStep: number;
  scaleUpCooldown: number;
  scaleDownCooldown: number;
}

// Cost Analysis
interface CostAnalysis {
  id: string;
  period: {
    start: Date;
    end: Date;
  };
  totalCost: number;
  costByResource: CostByResource[];
  costByType: Record<ResourceType, number>;
  costByProvider: Record<string, number>;
  costByTeam: Record<string, number>;
  costTrend: CostTrendPoint[];
  optimization: CostOptimization;
  forecast: CostForecast;
}

// Cost By Resource
interface CostByResource {
  resourceId: string;
  resourceName: string;
  resourceType: ResourceType;
  cost: number;
  costPerUnit: number;
  utilization: number;
  efficiency: number;
  wastedCost: number;
}

// Cost Trend Point
interface CostTrendPoint {
  date: Date;
  cost: number;
  budget: number;
  variance: number;
}

// Cost Optimization
interface CostOptimization {
  totalSavingsOpportunity: number;
  recommendations: CostRecommendation[];
  implementedSavings: number;
  pendingSavings: number;
}

// Cost Recommendation
interface CostRecommendation {
  id: string;
  type: 'rightsizing' | 'reservation' | 'spot' | 'unused' | 'scheduling' | 'tiering';
  resource: string;
  currentCost: number;
  projectedCost: number;
  savings: number;
  savingsPercent: number;
  effort: 'low' | 'medium' | 'high';
  risk: 'low' | 'medium' | 'high';
  status: 'identified' | 'approved' | 'implemented' | 'rejected';
}

// Cost Forecast
interface CostForecast {
  horizon: number;
  predictions: CostPrediction[];
  confidence: ForecastConfidence;
  annualProjection: number;
}

// Cost Prediction
interface CostPrediction {
  date: Date;
  predicted: number;
  lowerBound: number;
  upperBound: number;
}

// Capacity Statistics
interface CapacityStatistics {
  overview: {
    totalResources: number;
    healthyResources: number;
    warningResources: number;
    criticalResources: number;
    avgUtilization: number;
    totalCapacity: number;
    usedCapacity: number;
    availableCapacity: number;
  };
  byType: Record<ResourceType, ResourceTypeStats>;
  byProvider: Record<string, number>;
  byStatus: Record<CapacityStatus, number>;
  forecasting: {
    resourcesAtRisk: number;
    avgDaysToExhaustion: number;
    forecastAccuracy: number;
    totalPredictions: number;
  };
  scaling: {
    scaleUpEvents: number;
    scaleDownEvents: number;
    autoScalingEnabled: number;
    avgScalingLatency: number;
  };
  cost: {
    totalCost: number;
    projectedCost: number;
    savingsOpportunity: number;
    costEfficiency: number;
  };
}

// Resource Type Stats
interface ResourceTypeStats {
  count: number;
  totalCapacity: number;
  usedCapacity: number;
  avgUtilization: number;
  cost: number;
}

class CapacityPlanningService {
  private static instance: CapacityPlanningService;
  private resources: Map<string, Resource> = new Map();
  private plans: Map<string, CapacityPlan> = new Map();
  private alerts: Map<string, CapacityAlert> = new Map();
  private policies: Map<string, ScalingPolicy> = new Map();
  private costAnalyses: Map<string, CostAnalysis> = new Map();
  private eventListeners: ((event: string, data: unknown) => void)[] = [];

  private constructor() {
    this.initializeSampleData();
  }

  public static getInstance(): CapacityPlanningService {
    if (!CapacityPlanningService.instance) {
      CapacityPlanningService.instance = new CapacityPlanningService();
    }
    return CapacityPlanningService.instance;
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private initializeSampleData(): void {
    // Initialize Resources
    const resourcesData = [
      { name: 'API Gateway Cluster', type: 'compute' as ResourceType, capacity: 100, used: 65 },
      { name: 'User Service Pods', type: 'compute' as ResourceType, capacity: 50, used: 40 },
      { name: 'PostgreSQL Primary', type: 'database' as ResourceType, capacity: 500, used: 350 },
      { name: 'Redis Cache Cluster', type: 'cache' as ResourceType, capacity: 64, used: 45 },
      { name: 'S3 Storage Bucket', type: 'storage' as ResourceType, capacity: 10000, used: 6500 },
      { name: 'Kafka Message Queue', type: 'queue' as ResourceType, capacity: 1000000, used: 450000 },
      { name: 'Application Memory Pool', type: 'memory' as ResourceType, capacity: 256, used: 180 },
      { name: 'Network Bandwidth', type: 'network' as ResourceType, capacity: 10, used: 4.5 },
    ];

    resourcesData.forEach((r, idx) => {
      const utilization = (r.used / r.capacity) * 100;
      const resource: Resource = {
        id: `resource-${(idx + 1).toString().padStart(4, '0')}`,
        name: r.name,
        type: r.type,
        description: `${r.name} capacity tracking`,
        provider: {
          type: idx < 4 ? 'kubernetes' : idx < 6 ? 'aws' : 'on-premise',
          region: 'us-east-1',
          zone: 'us-east-1a',
          cluster: idx < 4 ? 'prod-cluster' : undefined,
          namespace: idx < 4 ? 'production' : undefined,
        },
        configuration: {
          instanceType: idx < 2 ? 'm5.xlarge' : undefined,
          vcpus: idx < 2 ? 4 : undefined,
          memoryGB: idx === 6 ? 256 : idx < 2 ? 16 : undefined,
          storageGB: idx === 4 ? 10000 : idx === 2 ? 500 : undefined,
          connections: idx === 2 ? 1000 : undefined,
          replicas: idx < 4 ? 3 : undefined,
        },
        capacity: {
          total: r.capacity,
          available: r.capacity - r.used,
          reserved: r.capacity * 0.1,
          used: r.used,
          unit: idx === 4 ? 'GB' : idx === 5 ? 'messages/s' : idx === 7 ? 'Gbps' : idx === 6 ? 'GB' : 'units',
          scalable: true,
          minCapacity: r.capacity * 0.2,
          maxCapacity: r.capacity * 3,
          currentCost: Math.floor(Math.random() * 5000) + 1000,
          costPerUnit: Math.random() * 10 + 1,
          currency: 'USD',
        },
        utilization: {
          current: utilization,
          average: utilization - 5 + Math.random() * 10,
          peak: utilization + 15,
          percentile95: utilization + 10,
          percentile99: utilization + 20,
          trend: utilization > 70 ? 'increasing' : utilization < 40 ? 'decreasing' : 'stable',
          trendPercentage: Math.random() * 10 - 5,
          history: Array.from({ length: 24 }, (_, i) => ({
            timestamp: new Date(Date.now() - i * 60 * 60 * 1000),
            value: utilization - 10 + Math.random() * 20,
            anomaly: Math.random() < 0.05,
          })),
          lastUpdated: new Date(),
        },
        thresholds: {
          warning: 70,
          critical: 85,
          targetUtilization: 60,
          minUtilization: 20,
          maxUtilization: 80,
          scaleUpThreshold: 75,
          scaleDownThreshold: 30,
          cooldownPeriod: 300,
        },
        forecast: {
          horizonDays: 30,
          predictions: Array.from({ length: 30 }, (_, i) => ({
            date: new Date(Date.now() + (i + 1) * 24 * 60 * 60 * 1000),
            predicted: utilization + i * 0.5,
            lowerBound: utilization + i * 0.3,
            upperBound: utilization + i * 0.7,
            confidence: 0.95 - i * 0.01,
          })),
          exhaustionDate: utilization > 60 ? new Date(Date.now() + Math.floor((100 - utilization) / 0.5) * 24 * 60 * 60 * 1000) : undefined,
          confidence: utilization > 70 ? 'high' : 'medium',
          model: 'arima',
          accuracy: 0.92,
          lastUpdated: new Date(),
        },
        recommendations: utilization > 70 ? [
          {
            id: `rec-${idx}-1`,
            type: 'scale',
            action: 'scale_out',
            priority: utilization > 85 ? 'critical' : 'high',
            title: `Scale out ${r.name}`,
            description: `Current utilization at ${utilization.toFixed(1)}%, recommend scaling out`,
            impact: {
              capacityChange: r.capacity * 0.25,
              costChange: 500,
              costChangePercent: 15,
              performanceImpact: 'positive',
              riskLevel: 'low',
              savingsPerMonth: -500,
            },
            implementation: {
              automated: true,
              steps: ['Trigger auto-scaling policy', 'Verify new instances', 'Update load balancer'],
              estimatedDuration: 10,
              downtime: false,
              approvalRequired: utilization < 85,
              rollbackAvailable: true,
            },
            status: 'pending',
            createdAt: new Date(),
            expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
          },
        ] : [],
        tags: [r.type, 'production', 'monitored'],
        labels: { environment: 'production', team: 'platform', tier: idx < 4 ? 'tier-1' : 'tier-2' },
        metadata: {
          createdAt: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000),
          createdBy: 'admin',
          updatedAt: new Date(),
          lastCapacityChange: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          version: 5,
        },
      };
      this.resources.set(resource.id, resource);
    });

    // Initialize Capacity Plan
    const plan: CapacityPlan = {
      id: 'plan-0001',
      name: 'Q1 2024 Capacity Plan',
      description: 'Quarterly capacity planning for production infrastructure',
      scope: {
        services: ['api-gateway', 'user-service', 'payment-service'],
        environments: ['production'],
        regions: ['us-east-1', 'eu-west-1'],
        resourceTypes: ['compute', 'database', 'cache', 'storage'],
        teams: ['platform', 'infrastructure'],
      },
      timeframe: {
        startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        endDate: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000),
        reviewCycle: 'monthly',
        nextReview: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
      objectives: [
        { id: 'obj-1', name: 'Maintain 99.9% availability', description: 'Ensure sufficient capacity for high availability', type: 'reliability', target: 99.9, current: 99.85, unit: '%', priority: 'critical', status: 'at_risk' },
        { id: 'obj-2', name: 'Reduce infrastructure cost', description: 'Optimize resource utilization', type: 'cost', target: 15, current: 8, unit: '% reduction', priority: 'high', status: 'on_track' },
        { id: 'obj-3', name: 'Support 50% traffic growth', description: 'Scale infrastructure for projected growth', type: 'growth', target: 50, current: 30, unit: '% headroom', priority: 'high', status: 'on_track' },
      ],
      resources: [
        {
          resourceId: 'resource-0001',
          resourceName: 'API Gateway Cluster',
          resourceType: 'compute',
          currentCapacity: 100,
          plannedCapacity: 150,
          requiredCapacity: 130,
          timeline: Array.from({ length: 3 }, (_, i) => ({
            date: new Date(Date.now() + i * 30 * 24 * 60 * 60 * 1000),
            plannedCapacity: 100 + i * 25,
            forecastedDemand: 65 + i * 20,
            buffer: 35 + i * 5,
          })),
          actions: [
            { id: 'action-1', action: 'scale_out', scheduledDate: new Date(Date.now() + 15 * 24 * 60 * 60 * 1000), capacityChange: 25, cost: 2500, status: 'planned' },
          ],
        },
      ],
      scenarios: [
        {
          id: 'scenario-1',
          name: 'Baseline',
          description: 'Expected growth scenario',
          type: 'baseline',
          assumptions: [{ name: 'Traffic Growth', value: 10, unit: '%/month', description: 'Monthly traffic increase' }],
          projections: Array.from({ length: 3 }, (_, i) => ({
            month: i + 1,
            demand: 65 + i * 6.5,
            capacity: 100 + i * 25,
            cost: 10000 + i * 2500,
            utilizationPercent: (65 + i * 6.5) / (100 + i * 25) * 100,
          })),
          budget: 40000,
          likelihood: 0.7,
        },
        {
          id: 'scenario-2',
          name: 'High Growth',
          description: 'Optimistic growth scenario',
          type: 'optimistic',
          assumptions: [{ name: 'Traffic Growth', value: 20, unit: '%/month', description: 'Monthly traffic increase' }],
          projections: Array.from({ length: 3 }, (_, i) => ({
            month: i + 1,
            demand: 65 + i * 13,
            capacity: 100 + i * 35,
            cost: 10000 + i * 3500,
            utilizationPercent: (65 + i * 13) / (100 + i * 35) * 100,
          })),
          budget: 50000,
          likelihood: 0.2,
        },
      ],
      budgets: [
        { id: 'budget-1', category: 'compute', allocated: 25000, spent: 18000, forecast: 24000, variance: -1000, currency: 'USD' },
        { id: 'budget-2', category: 'storage', allocated: 10000, spent: 7500, forecast: 9500, variance: -500, currency: 'USD' },
        { id: 'budget-3', category: 'network', allocated: 5000, spent: 3500, forecast: 4800, variance: -200, currency: 'USD' },
      ],
      risks: [
        { id: 'risk-1', name: 'Traffic Spike', description: 'Unexpected traffic spike during peak season', category: 'capacity', likelihood: 'medium', impact: 'high', mitigations: ['Auto-scaling policies', 'CDN caching'], owner: 'platform-team', status: 'mitigated' },
        { id: 'risk-2', name: 'Budget Overrun', description: 'Exceeding quarterly budget', category: 'budget', likelihood: 'low', impact: 'medium', mitigations: ['Cost alerts', 'Reserved instances'], owner: 'finance', status: 'identified' },
      ],
      milestones: [
        { id: 'milestone-1', name: 'Scale API Gateway', description: 'Complete API gateway scaling', targetDate: new Date(Date.now() + 15 * 24 * 60 * 60 * 1000), status: 'pending', deliverables: ['Additional instances', 'Load balancer configuration'] },
        { id: 'milestone-2', name: 'Database Upgrade', description: 'Upgrade database capacity', targetDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), status: 'pending', deliverables: ['New database instance', 'Migration completed'] },
      ],
      status: 'active',
      approvals: [
        { approver: 'VP Engineering', role: 'executive', status: 'approved', date: new Date(Date.now() - 25 * 24 * 60 * 60 * 1000), comments: 'Approved with budget constraint' },
        { approver: 'Finance Director', role: 'finance', status: 'approved', date: new Date(Date.now() - 24 * 24 * 60 * 60 * 1000) },
      ],
      metadata: {
        createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        createdBy: 'capacity-manager',
        updatedAt: new Date(),
        version: 3,
      },
    };
    this.plans.set(plan.id, plan);

    // Initialize Capacity Alerts
    const alertsData = [
      { resource: 'resource-0001', type: 'threshold' as const, severity: 'warning' as const, title: 'High CPU utilization' },
      { resource: 'resource-0003', type: 'forecast' as const, severity: 'warning' as const, title: 'Database capacity exhaustion predicted' },
    ];

    alertsData.forEach((a, idx) => {
      const alert: CapacityAlert = {
        id: `alert-${(idx + 1).toString().padStart(4, '0')}`,
        resourceId: a.resource,
        resourceName: resourcesData[parseInt(a.resource.split('-')[1]) - 1].name,
        resourceType: resourcesData[parseInt(a.resource.split('-')[1]) - 1].type,
        alertType: a.type,
        severity: a.severity,
        title: a.title,
        message: `${a.title} on ${resourcesData[parseInt(a.resource.split('-')[1]) - 1].name}`,
        currentValue: 75 + idx * 5,
        thresholdValue: 70,
        triggeredAt: new Date(Date.now() - idx * 60 * 60 * 1000),
        status: idx === 0 ? 'active' : 'acknowledged',
        acknowledgedAt: idx === 1 ? new Date() : undefined,
        acknowledgedBy: idx === 1 ? 'admin' : undefined,
        actions: [
          { type: 'notify', target: 'platform-team', status: 'completed', executedAt: new Date() },
        ],
      };
      this.alerts.set(alert.id, alert);
    });

    // Initialize Scaling Policies
    const scalingPoliciesData = [
      { name: 'API Gateway Auto-Scale', resourceType: 'compute' as ResourceType },
      { name: 'Database Connection Pool', resourceType: 'database' as ResourceType },
    ];

    scalingPoliciesData.forEach((p, idx) => {
      const policy: ScalingPolicy = {
        id: `policy-${(idx + 1).toString().padStart(4, '0')}`,
        name: p.name,
        description: `Auto-scaling policy for ${p.name}`,
        resourceType: p.resourceType,
        resources: [`resource-${(idx + 1).toString().padStart(4, '0')}`],
        triggers: [
          { id: `trigger-${idx}-1`, metric: 'cpu_utilization', condition: 'above', threshold: 75, duration: 300, evaluationPeriods: 3 },
          { id: `trigger-${idx}-2`, metric: 'cpu_utilization', condition: 'below', threshold: 30, duration: 600, evaluationPeriods: 5 },
        ],
        actions: [
          { type: 'scale_out', adjustment: 2, adjustmentType: 'absolute', minCapacity: 2, maxCapacity: 20 },
          { type: 'scale_in', adjustment: 1, adjustmentType: 'absolute', minCapacity: 2, maxCapacity: 20 },
        ],
        cooldown: 300,
        schedule: {
          enabled: true,
          timezone: 'America/New_York',
          schedules: [
            { id: 'sched-1', name: 'Business Hours Scale Up', cron: '0 8 * * 1-5', targetCapacity: 10, enabled: true },
            { id: 'sched-2', name: 'Night Scale Down', cron: '0 20 * * 1-5', targetCapacity: 5, enabled: true },
          ],
        },
        limits: {
          minCapacity: 2,
          maxCapacity: 20,
          maxScaleUpStep: 5,
          maxScaleDownStep: 2,
          scaleUpCooldown: 300,
          scaleDownCooldown: 600,
        },
        status: 'active',
        metadata: {
          createdAt: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000),
          createdBy: 'admin',
          updatedAt: new Date(),
          lastTriggered: new Date(Date.now() - 2 * 60 * 60 * 1000),
          triggerCount: 150,
        },
      };
      this.policies.set(policy.id, policy);
    });

    // Initialize Cost Analysis
    const costAnalysis: CostAnalysis = {
      id: 'cost-0001',
      period: {
        start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        end: new Date(),
      },
      totalCost: 45000,
      costByResource: resourcesData.map((r, idx) => ({
        resourceId: `resource-${(idx + 1).toString().padStart(4, '0')}`,
        resourceName: r.name,
        resourceType: r.type,
        cost: 3000 + Math.random() * 5000,
        costPerUnit: Math.random() * 10 + 1,
        utilization: (r.used / r.capacity) * 100,
        efficiency: 70 + Math.random() * 25,
        wastedCost: Math.random() * 500,
      })),
      costByType: { compute: 20000, database: 10000, storage: 8000, cache: 4000, network: 2000, memory: 500, queue: 500, custom: 0 },
      costByProvider: { aws: 30000, kubernetes: 10000, 'on-premise': 5000 },
      costByTeam: { platform: 25000, infrastructure: 15000, application: 5000 },
      costTrend: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000),
        cost: 1400 + Math.random() * 200,
        budget: 1500,
        variance: Math.random() * 200 - 100,
      })),
      optimization: {
        totalSavingsOpportunity: 8500,
        recommendations: [
          { id: 'cost-rec-1', type: 'rightsizing', resource: 'API Gateway Cluster', currentCost: 5000, projectedCost: 3500, savings: 1500, savingsPercent: 30, effort: 'low', risk: 'low', status: 'identified' },
          { id: 'cost-rec-2', type: 'reservation', resource: 'PostgreSQL Primary', currentCost: 4000, projectedCost: 2800, savings: 1200, savingsPercent: 30, effort: 'medium', risk: 'low', status: 'approved' },
          { id: 'cost-rec-3', type: 'spot', resource: 'User Service Pods', currentCost: 3000, projectedCost: 900, savings: 2100, savingsPercent: 70, effort: 'high', risk: 'medium', status: 'identified' },
        ],
        implementedSavings: 2000,
        pendingSavings: 6500,
      },
      forecast: {
        horizon: 90,
        predictions: Array.from({ length: 3 }, (_, i) => ({
          date: new Date(Date.now() + (i + 1) * 30 * 24 * 60 * 60 * 1000),
          predicted: 45000 + i * 2000,
          lowerBound: 43000 + i * 1500,
          upperBound: 47000 + i * 2500,
        })),
        confidence: 'medium',
        annualProjection: 550000,
      },
    };
    this.costAnalyses.set(costAnalysis.id, costAnalysis);
  }

  // Resource Operations
  public getResources(type?: ResourceType): Resource[] {
    let resources = Array.from(this.resources.values());
    if (type) resources = resources.filter((r) => r.type === type);
    return resources;
  }

  public getResourceById(id: string): Resource | undefined {
    return this.resources.get(id);
  }

  public getResourcesByStatus(status: CapacityStatus): Resource[] {
    return Array.from(this.resources.values()).filter((r) => {
      const util = r.utilization.current;
      if (status === 'critical') return util >= r.thresholds.critical;
      if (status === 'warning') return util >= r.thresholds.warning && util < r.thresholds.critical;
      if (status === 'healthy') return util < r.thresholds.warning;
      return false;
    });
  }

  // Plan Operations
  public getPlans(): CapacityPlan[] {
    return Array.from(this.plans.values());
  }

  public getPlanById(id: string): CapacityPlan | undefined {
    return this.plans.get(id);
  }

  // Alert Operations
  public getAlerts(status?: 'active' | 'acknowledged' | 'resolved'): CapacityAlert[] {
    let alerts = Array.from(this.alerts.values());
    if (status) alerts = alerts.filter((a) => a.status === status);
    return alerts.sort((a, b) => b.triggeredAt.getTime() - a.triggeredAt.getTime());
  }

  public getAlertById(id: string): CapacityAlert | undefined {
    return this.alerts.get(id);
  }

  // Policy Operations
  public getPolicies(): ScalingPolicy[] {
    return Array.from(this.policies.values());
  }

  public getPolicyById(id: string): ScalingPolicy | undefined {
    return this.policies.get(id);
  }

  // Cost Operations
  public getCostAnalyses(): CostAnalysis[] {
    return Array.from(this.costAnalyses.values());
  }

  public getLatestCostAnalysis(): CostAnalysis | undefined {
    const analyses = Array.from(this.costAnalyses.values());
    return analyses.sort((a, b) => b.period.end.getTime() - a.period.end.getTime())[0];
  }

  // Statistics
  public getStatistics(): CapacityStatistics {
    const resources = Array.from(this.resources.values());
    const alerts = Array.from(this.alerts.values());
    const policies = Array.from(this.policies.values());
    const costAnalysis = this.getLatestCostAnalysis();

    const byType: Record<ResourceType, ResourceTypeStats> = {
      compute: { count: 0, totalCapacity: 0, usedCapacity: 0, avgUtilization: 0, cost: 0 },
      memory: { count: 0, totalCapacity: 0, usedCapacity: 0, avgUtilization: 0, cost: 0 },
      storage: { count: 0, totalCapacity: 0, usedCapacity: 0, avgUtilization: 0, cost: 0 },
      network: { count: 0, totalCapacity: 0, usedCapacity: 0, avgUtilization: 0, cost: 0 },
      database: { count: 0, totalCapacity: 0, usedCapacity: 0, avgUtilization: 0, cost: 0 },
      cache: { count: 0, totalCapacity: 0, usedCapacity: 0, avgUtilization: 0, cost: 0 },
      queue: { count: 0, totalCapacity: 0, usedCapacity: 0, avgUtilization: 0, cost: 0 },
      custom: { count: 0, totalCapacity: 0, usedCapacity: 0, avgUtilization: 0, cost: 0 },
    };

    const byProvider: Record<string, number> = {};
    const byStatus: Record<CapacityStatus, number> = { healthy: 0, warning: 0, critical: 0, exhausted: 0, unknown: 0 };

    resources.forEach((r) => {
      byType[r.type].count++;
      byType[r.type].totalCapacity += r.capacity.total;
      byType[r.type].usedCapacity += r.capacity.used;
      byType[r.type].cost += r.capacity.currentCost;

      byProvider[r.provider.type] = (byProvider[r.provider.type] || 0) + 1;

      const util = r.utilization.current;
      if (util >= r.thresholds.critical) byStatus.critical++;
      else if (util >= r.thresholds.warning) byStatus.warning++;
      else byStatus.healthy++;
    });

    Object.keys(byType).forEach((type) => {
      const t = type as ResourceType;
      if (byType[t].count > 0) {
        byType[t].avgUtilization = (byType[t].usedCapacity / byType[t].totalCapacity) * 100;
      }
    });

    return {
      overview: {
        totalResources: resources.length,
        healthyResources: byStatus.healthy,
        warningResources: byStatus.warning,
        criticalResources: byStatus.critical,
        avgUtilization: resources.reduce((sum, r) => sum + r.utilization.current, 0) / resources.length,
        totalCapacity: resources.reduce((sum, r) => sum + r.capacity.total, 0),
        usedCapacity: resources.reduce((sum, r) => sum + r.capacity.used, 0),
        availableCapacity: resources.reduce((sum, r) => sum + r.capacity.available, 0),
      },
      byType,
      byProvider,
      byStatus,
      forecasting: {
        resourcesAtRisk: resources.filter((r) => r.forecast.exhaustionDate && r.forecast.exhaustionDate < new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)).length,
        avgDaysToExhaustion: resources.filter((r) => r.forecast.exhaustionDate).reduce((sum, r) => sum + (r.forecast.exhaustionDate!.getTime() - Date.now()) / (24 * 60 * 60 * 1000), 0) / (resources.filter((r) => r.forecast.exhaustionDate).length || 1),
        forecastAccuracy: resources.reduce((sum, r) => sum + r.forecast.accuracy, 0) / resources.length,
        totalPredictions: resources.reduce((sum, r) => sum + r.forecast.predictions.length, 0),
      },
      scaling: {
        scaleUpEvents: policies.reduce((sum, p) => sum + Math.floor(p.metadata.triggerCount * 0.6), 0),
        scaleDownEvents: policies.reduce((sum, p) => sum + Math.floor(p.metadata.triggerCount * 0.4), 0),
        autoScalingEnabled: policies.filter((p) => p.status === 'active').length,
        avgScalingLatency: 45,
      },
      cost: {
        totalCost: costAnalysis?.totalCost || 0,
        projectedCost: costAnalysis?.forecast.annualProjection || 0,
        savingsOpportunity: costAnalysis?.optimization.totalSavingsOpportunity || 0,
        costEfficiency: costAnalysis ? (costAnalysis.costByResource.reduce((sum, r) => sum + r.efficiency, 0) / costAnalysis.costByResource.length) : 0,
      },
    };
  }

  // Event Handling
  public subscribe(callback: (event: string, data: unknown) => void): () => void {
    this.eventListeners.push(callback);
    return () => {
      const index = this.eventListeners.indexOf(callback);
      if (index > -1) this.eventListeners.splice(index, 1);
    };
  }

  private emit(event: string, data: unknown): void {
    this.eventListeners.forEach((callback) => callback(event, data));
  }
}

export const capacityPlanningService = CapacityPlanningService.getInstance();
export type {
  ResourceType,
  CapacityStatus,
  TrendDirection,
  ForecastConfidence,
  ScalingAction,
  Resource,
  ResourceProvider,
  ResourceConfiguration,
  ResourceCapacity,
  ResourceUtilization,
  UtilizationDataPoint,
  ResourceThresholds,
  ResourceForecast,
  ForecastPrediction,
  ResourceRecommendation,
  RecommendationImpact,
  RecommendationImplementation,
  ResourceMetadata,
  CapacityPlan,
  PlanScope,
  PlanTimeframe,
  PlanObjective,
  PlanResource,
  CapacityTimeline,
  PlannedAction,
  CapacityScenario,
  ScenarioAssumption,
  ScenarioProjection,
  PlanBudget,
  PlanRisk,
  PlanMilestone,
  PlanApproval,
  CapacityAlert,
  AlertAction,
  ScalingPolicy,
  ScalingTrigger,
  ScalingPolicyAction,
  ScalingSchedule,
  ScheduledScaling,
  ScalingLimits,
  CostAnalysis,
  CostByResource,
  CostTrendPoint,
  CostOptimization,
  CostRecommendation,
  CostForecast,
  CostPrediction,
  CapacityStatistics,
  ResourceTypeStats,
};
