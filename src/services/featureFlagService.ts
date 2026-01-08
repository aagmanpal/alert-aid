/**
 * Feature Flag Service
 * Comprehensive feature flag management, targeting, and experimentation
 */

// Flag Type
type FlagType = 'boolean' | 'string' | 'number' | 'json' | 'multivariate';

// Flag Status
type FlagStatus = 'active' | 'inactive' | 'archived' | 'scheduled';

// Rollout Strategy
type RolloutStrategy = 'percentage' | 'user_segment' | 'gradual' | 'ring' | 'random';

// Targeting Operator
type TargetingOperator = 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'starts_with' | 'ends_with' | 'matches' | 'in' | 'not_in' | 'gt' | 'lt' | 'gte' | 'lte' | 'exists' | 'not_exists';

// Feature Flag
interface FeatureFlag {
  id: string;
  key: string;
  name: string;
  description: string;
  type: FlagType;
  project: string;
  environment: string;
  defaultValue: FlagValue;
  variations: FlagVariation[];
  targeting: FlagTargeting;
  scheduling: FlagScheduling;
  rollout: FlagRollout;
  prerequisites: FlagPrerequisite[];
  metrics: FlagMetrics;
  audit: FlagAudit;
  ownership: FlagOwnership;
  status: FlagStatus;
  metadata: FlagMetadata;
}

// Flag Value
interface FlagValue {
  type: FlagType;
  booleanValue?: boolean;
  stringValue?: string;
  numberValue?: number;
  jsonValue?: Record<string, unknown>;
}

// Flag Variation
interface FlagVariation {
  id: string;
  name: string;
  description?: string;
  value: FlagValue;
  weight?: number;
  isControl?: boolean;
}

// Flag Targeting
interface FlagTargeting {
  enabled: boolean;
  rules: TargetingRule[];
  defaultVariation: string;
  offVariation?: string;
  userTargeting: UserTargeting;
  contextTargeting: ContextTargeting;
}

// Targeting Rule
interface TargetingRule {
  id: string;
  name: string;
  description?: string;
  priority: number;
  conditions: TargetingCondition[];
  variation: string;
  percentage?: number;
  bucketBy?: string;
  enabled: boolean;
}

// Targeting Condition
interface TargetingCondition {
  attribute: string;
  operator: TargetingOperator;
  value: unknown;
  values?: unknown[];
  negate?: boolean;
}

// User Targeting
interface UserTargeting {
  enabled: boolean;
  includedUsers: string[];
  excludedUsers: string[];
  userGroups: UserGroup[];
}

// User Group
interface UserGroup {
  id: string;
  name: string;
  description?: string;
  members: string[];
  variation: string;
}

// Context Targeting
interface ContextTargeting {
  enabled: boolean;
  contexts: ContextRule[];
}

// Context Rule
interface ContextRule {
  id: string;
  name: string;
  contextKind: string;
  conditions: TargetingCondition[];
  variation: string;
}

// Flag Scheduling
interface FlagScheduling {
  enabled: boolean;
  schedules: FlagSchedule[];
  timezone: string;
}

// Flag Schedule
interface FlagSchedule {
  id: string;
  name: string;
  action: 'enable' | 'disable' | 'update';
  startTime: Date;
  endTime?: Date;
  recurrence?: ScheduleRecurrence;
  variation?: string;
  executed: boolean;
  executedAt?: Date;
}

// Schedule Recurrence
interface ScheduleRecurrence {
  frequency: 'daily' | 'weekly' | 'monthly' | 'yearly';
  interval: number;
  daysOfWeek?: number[];
  daysOfMonth?: number[];
  endAfter?: number;
  endDate?: Date;
}

// Flag Rollout
interface FlagRollout {
  strategy: RolloutStrategy;
  percentage: number;
  stages: RolloutStage[];
  currentStage?: number;
  bucketBy: string;
  seed?: number;
  gradualRollout?: GradualRollout;
  ringRollout?: RingRollout;
}

// Rollout Stage
interface RolloutStage {
  id: string;
  name: string;
  percentage: number;
  duration: number;
  criteria?: RolloutCriteria;
  status: 'pending' | 'active' | 'completed' | 'failed';
  startedAt?: Date;
  completedAt?: Date;
}

// Rollout Criteria
interface RolloutCriteria {
  minSampleSize: number;
  maxErrorRate: number;
  maxLatencyP95: number;
  customMetrics?: MetricCriteria[];
}

// Metric Criteria
interface MetricCriteria {
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte';
  threshold: number;
}

// Gradual Rollout
interface GradualRollout {
  initialPercentage: number;
  increment: number;
  intervalMinutes: number;
  targetPercentage: number;
  currentPercentage: number;
  pauseOnError: boolean;
  lastIncrement?: Date;
}

// Ring Rollout
interface RingRollout {
  rings: RolloutRing[];
  currentRing: number;
  autoPromote: boolean;
  promotionDelay: number;
}

// Rollout Ring
interface RolloutRing {
  id: string;
  name: string;
  description?: string;
  percentage: number;
  targeting?: TargetingRule[];
  status: 'pending' | 'active' | 'completed';
  promotedAt?: Date;
}

// Flag Prerequisite
interface FlagPrerequisite {
  flagKey: string;
  variation: string;
}

// Flag Metrics
interface FlagMetrics {
  enabled: boolean;
  goals: FlagGoal[];
  experiments: FlagExperiment[];
  currentExperiment?: string;
}

// Flag Goal
interface FlagGoal {
  id: string;
  name: string;
  description?: string;
  type: 'conversion' | 'numeric' | 'custom';
  event: string;
  successCriteria?: GoalCriteria;
  baseline?: number;
  target?: number;
}

// Goal Criteria
interface GoalCriteria {
  minEffect: number;
  minConfidence: number;
  direction: 'increase' | 'decrease' | 'any';
}

// Flag Experiment
interface FlagExperiment {
  id: string;
  name: string;
  description?: string;
  hypothesis?: string;
  status: 'draft' | 'running' | 'paused' | 'completed' | 'cancelled';
  variations: ExperimentVariation[];
  traffic: ExperimentTraffic;
  metrics: ExperimentMetric[];
  results?: ExperimentResults;
  scheduling: ExperimentScheduling;
  metadata: ExperimentMetadata;
}

// Experiment Variation
interface ExperimentVariation {
  variationId: string;
  name: string;
  weight: number;
  isControl: boolean;
}

// Experiment Traffic
interface ExperimentTraffic {
  allocation: number;
  samplingRate: number;
  bucketBy: string;
  seed?: number;
}

// Experiment Metric
interface ExperimentMetric {
  goalId: string;
  name: string;
  isPrimary: boolean;
  minimumSampleSize: number;
  minimumDetectableEffect: number;
}

// Experiment Results
interface ExperimentResults {
  startDate: Date;
  endDate?: Date;
  totalParticipants: number;
  variationResults: VariationResult[];
  winner?: string;
  significance: number;
  confidence: number;
  recommendation?: string;
}

// Variation Result
interface VariationResult {
  variationId: string;
  participants: number;
  conversions: number;
  conversionRate: number;
  improvement?: number;
  confidence?: number;
  pValue?: number;
  isWinner: boolean;
}

// Experiment Scheduling
interface ExperimentScheduling {
  startDate?: Date;
  endDate?: Date;
  minDuration: number;
  maxDuration?: number;
  autoStop: boolean;
  stopCriteria?: StopCriteria;
}

// Stop Criteria
interface StopCriteria {
  minSampleSize: number;
  minSignificance: number;
  maxPValue: number;
}

// Experiment Metadata
interface ExperimentMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  tags: string[];
}

// Flag Audit
interface FlagAudit {
  enabled: boolean;
  events: FlagAuditEvent[];
  retentionDays: number;
}

// Flag Audit Event
interface FlagAuditEvent {
  id: string;
  timestamp: Date;
  action: 'created' | 'updated' | 'enabled' | 'disabled' | 'archived' | 'targeting_changed' | 'rollout_changed' | 'experiment_started' | 'experiment_stopped';
  actor: string;
  actorType: 'user' | 'service' | 'system' | 'schedule';
  details: Record<string, unknown>;
  previousValue?: unknown;
  newValue?: unknown;
}

// Flag Ownership
interface FlagOwnership {
  owner: string;
  team: string;
  maintainers: string[];
  reviewers: string[];
  stakeholders: string[];
}

// Flag Metadata
interface FlagMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  updatedBy: string;
  version: number;
  tags: string[];
  labels: Record<string, string>;
  temporary: boolean;
  expirationDate?: Date;
  jiraTicket?: string;
  documentationUrl?: string;
}

// Flag Project
interface FlagProject {
  id: string;
  key: string;
  name: string;
  description: string;
  environments: ProjectEnvironment[];
  defaultEnvironment: string;
  flags: string[];
  segments: string[];
  settings: ProjectSettings;
  access: ProjectAccess;
  integrations: ProjectIntegration[];
  metadata: ProjectMetadata;
}

// Project Environment
interface ProjectEnvironment {
  id: string;
  key: string;
  name: string;
  color: string;
  type: 'development' | 'staging' | 'production' | 'testing';
  apiKey: string;
  clientSideId: string;
  mobileKey?: string;
  defaultTTL: number;
  secureMode: boolean;
  defaultTrackEvents: boolean;
  requireComments: boolean;
  confirmChanges: boolean;
}

// Project Settings
interface ProjectSettings {
  defaultClientSideAvailability: {
    usingEnvironmentId: boolean;
    usingMobileKey: boolean;
  };
  defaultRelayProxyMode: 'proxy' | 'daemon';
  requireApproval: boolean;
  approvalSettings?: ApprovalSettings;
}

// Approval Settings
interface ApprovalSettings {
  required: boolean;
  minApprovers: number;
  canReviewOwnRequest: boolean;
  autoApproveScheduled: boolean;
  serviceAccounts: string[];
}

// Project Access
interface ProjectAccess {
  owner: string;
  admins: string[];
  writers: string[];
  readers: string[];
  customRoles: AccessRole[];
}

// Access Role
interface AccessRole {
  id: string;
  name: string;
  permissions: string[];
  members: string[];
}

// Project Integration
interface ProjectIntegration {
  id: string;
  type: 'slack' | 'jira' | 'github' | 'datadog' | 'webhook' | 'custom';
  name: string;
  config: Record<string, unknown>;
  events: string[];
  enabled: boolean;
}

// Project Metadata
interface ProjectMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  tags: string[];
}

// User Segment
interface UserSegment {
  id: string;
  key: string;
  name: string;
  description?: string;
  project: string;
  rules: SegmentRule[];
  included: string[];
  excluded: string[];
  unbounded: boolean;
  generation: number;
  version: number;
  metadata: SegmentMetadata;
}

// Segment Rule
interface SegmentRule {
  id: string;
  clauses: SegmentClause[];
  weight?: number;
  bucketBy?: string;
}

// Segment Clause
interface SegmentClause {
  attribute: string;
  operator: TargetingOperator;
  value: unknown;
  values?: unknown[];
  negate: boolean;
}

// Segment Metadata
interface SegmentMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  tags: string[];
}

// Flag Evaluation
interface FlagEvaluation {
  flagKey: string;
  value: FlagValue;
  variationIndex: number;
  variationId: string;
  reason: EvaluationReason;
  context: EvaluationContext;
  timestamp: Date;
  trackEvents: boolean;
}

// Evaluation Reason
interface EvaluationReason {
  kind: 'OFF' | 'FALLTHROUGH' | 'TARGET_MATCH' | 'RULE_MATCH' | 'PREREQUISITE_FAILED' | 'ERROR';
  ruleIndex?: number;
  ruleId?: string;
  prerequisiteKey?: string;
  errorKind?: string;
  inExperiment?: boolean;
}

// Evaluation Context
interface EvaluationContext {
  kind: string;
  key: string;
  name?: string;
  anonymous?: boolean;
  attributes: Record<string, unknown>;
  privateAttributes?: string[];
}

// Flag Change Request
interface FlagChangeRequest {
  id: string;
  flagKey: string;
  project: string;
  environment: string;
  requestor: ChangeRequestor;
  changes: FlagChange[];
  review: ChangeReview;
  status: 'pending' | 'approved' | 'rejected' | 'applied' | 'cancelled';
  scheduling?: ChangeScheduling;
  metadata: ChangeRequestMetadata;
}

// Change Requestor
interface ChangeRequestor {
  id: string;
  name: string;
  email: string;
  reason: string;
}

// Flag Change
interface FlagChange {
  field: string;
  previousValue: unknown;
  newValue: unknown;
  impact?: ChangeImpact;
}

// Change Impact
interface ChangeImpact {
  affectedUsers: number;
  affectedPercentage: number;
  riskLevel: 'low' | 'medium' | 'high';
}

// Change Review
interface ChangeReview {
  required: boolean;
  reviewers: ReviewerInfo[];
  approvals: ReviewApproval[];
  comments: ReviewComment[];
  status: 'pending' | 'approved' | 'rejected';
}

// Reviewer Info
interface ReviewerInfo {
  id: string;
  name: string;
  email: string;
  status: 'pending' | 'approved' | 'rejected';
}

// Review Approval
interface ReviewApproval {
  reviewerId: string;
  timestamp: Date;
  status: 'approved' | 'rejected';
  comment?: string;
}

// Review Comment
interface ReviewComment {
  id: string;
  author: string;
  timestamp: Date;
  content: string;
  resolved: boolean;
}

// Change Scheduling
interface ChangeScheduling {
  scheduledAt: Date;
  timezone: string;
  autoApply: boolean;
  notifyBefore: number;
}

// Change Request Metadata
interface ChangeRequestMetadata {
  createdAt: Date;
  updatedAt: Date;
  appliedAt?: Date;
  jiraTicket?: string;
}

// Feature Flag Statistics
interface FeatureFlagStatistics {
  overview: {
    totalFlags: number;
    activeFlags: number;
    inactiveFlags: number;
    archivedFlags: number;
    temporaryFlags: number;
    permanentFlags: number;
    expiringFlags: number;
  };
  byType: Record<FlagType, number>;
  byStatus: Record<FlagStatus, number>;
  byProject: Record<string, number>;
  byEnvironment: Record<string, number>;
  experiments: {
    active: number;
    completed: number;
    paused: number;
    avgDuration: number;
    winRate: number;
  };
  evaluations: {
    total: number;
    today: number;
    avgLatency: number;
    errorRate: number;
  };
  changes: {
    today: number;
    thisWeek: number;
    pendingApproval: number;
  };
}

class FeatureFlagService {
  private static instance: FeatureFlagService;
  private flags: Map<string, FeatureFlag> = new Map();
  private projects: Map<string, FlagProject> = new Map();
  private segments: Map<string, UserSegment> = new Map();
  private changeRequests: Map<string, FlagChangeRequest> = new Map();
  private eventListeners: ((event: string, data: unknown) => void)[] = [];

  private constructor() {
    this.initializeSampleData();
  }

  public static getInstance(): FeatureFlagService {
    if (!FeatureFlagService.instance) {
      FeatureFlagService.instance = new FeatureFlagService();
    }
    return FeatureFlagService.instance;
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private initializeSampleData(): void {
    // Initialize Projects
    const projectsData = [
      { key: 'web-app', name: 'Web Application' },
      { key: 'mobile-app', name: 'Mobile Application' },
      { key: 'api', name: 'API Services' },
    ];

    projectsData.forEach((proj, idx) => {
      const project: FlagProject = {
        id: `proj-${(idx + 1).toString().padStart(4, '0')}`,
        key: proj.key,
        name: proj.name,
        description: `${proj.name} feature flags`,
        environments: [
          { id: 'env-dev', key: 'development', name: 'Development', color: '#4CAF50', type: 'development', apiKey: `dev-${Math.random().toString(36).substr(2, 24)}`, clientSideId: `client-dev-${Math.random().toString(36).substr(2, 16)}`, defaultTTL: 300, secureMode: false, defaultTrackEvents: true, requireComments: false, confirmChanges: false },
          { id: 'env-stg', key: 'staging', name: 'Staging', color: '#FF9800', type: 'staging', apiKey: `stg-${Math.random().toString(36).substr(2, 24)}`, clientSideId: `client-stg-${Math.random().toString(36).substr(2, 16)}`, defaultTTL: 300, secureMode: false, defaultTrackEvents: true, requireComments: true, confirmChanges: true },
          { id: 'env-prod', key: 'production', name: 'Production', color: '#F44336', type: 'production', apiKey: `prod-${Math.random().toString(36).substr(2, 24)}`, clientSideId: `client-prod-${Math.random().toString(36).substr(2, 16)}`, defaultTTL: 60, secureMode: true, defaultTrackEvents: true, requireComments: true, confirmChanges: true },
        ],
        defaultEnvironment: 'production',
        flags: [],
        segments: [],
        settings: {
          defaultClientSideAvailability: { usingEnvironmentId: true, usingMobileKey: true },
          defaultRelayProxyMode: 'proxy',
          requireApproval: true,
          approvalSettings: { required: true, minApprovers: 1, canReviewOwnRequest: false, autoApproveScheduled: true, serviceAccounts: [] },
        },
        access: { owner: 'admin', admins: ['admin'], writers: ['developers'], readers: ['qa-team'], customRoles: [] },
        integrations: [
          { id: 'int-slack', type: 'slack', name: 'Slack Notifications', config: { channel: '#feature-flags' }, events: ['flag_enabled', 'flag_disabled', 'experiment_started'], enabled: true },
        ],
        metadata: { createdAt: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), createdBy: 'admin', updatedAt: new Date(), tags: [proj.key] },
      };
      this.projects.set(project.id, project);
    });

    // Initialize Segments
    const segmentsData = [
      { key: 'beta-users', name: 'Beta Users' },
      { key: 'premium-users', name: 'Premium Users' },
      { key: 'internal-users', name: 'Internal Users' },
    ];

    segmentsData.forEach((seg, idx) => {
      const segment: UserSegment = {
        id: `seg-${(idx + 1).toString().padStart(4, '0')}`,
        key: seg.key,
        name: seg.name,
        description: `${seg.name} segment`,
        project: 'proj-0001',
        rules: [
          { id: 'rule-1', clauses: [{ attribute: 'plan', operator: 'equals', value: idx === 1 ? 'premium' : 'beta', negate: false }] },
        ],
        included: idx === 2 ? ['user-admin', 'user-dev'] : [],
        excluded: [],
        unbounded: false,
        generation: 1,
        version: 1,
        metadata: { createdAt: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000), createdBy: 'admin', updatedAt: new Date(), tags: [seg.key] },
      };
      this.segments.set(segment.id, segment);
    });

    // Initialize Feature Flags
    const flagsData = [
      { key: 'new-dashboard', name: 'New Dashboard', type: 'boolean' as FlagType, status: 'active' as FlagStatus },
      { key: 'dark-mode', name: 'Dark Mode', type: 'boolean' as FlagType, status: 'active' as FlagStatus },
      { key: 'checkout-v2', name: 'Checkout V2', type: 'boolean' as FlagType, status: 'active' as FlagStatus },
      { key: 'search-algorithm', name: 'Search Algorithm', type: 'string' as FlagType, status: 'active' as FlagStatus },
      { key: 'max-items', name: 'Max Items Per Page', type: 'number' as FlagType, status: 'active' as FlagStatus },
      { key: 'pricing-experiment', name: 'Pricing Experiment', type: 'multivariate' as FlagType, status: 'active' as FlagStatus },
      { key: 'recommendations', name: 'Product Recommendations', type: 'boolean' as FlagType, status: 'inactive' as FlagStatus },
      { key: 'beta-features', name: 'Beta Features', type: 'json' as FlagType, status: 'active' as FlagStatus },
    ];

    flagsData.forEach((fg, idx) => {
      const flagId = `flag-${(idx + 1).toString().padStart(4, '0')}`;
      const variations: FlagVariation[] = fg.type === 'boolean' ? [
        { id: 'var-true', name: 'Enabled', value: { type: 'boolean', booleanValue: true }, isControl: false },
        { id: 'var-false', name: 'Disabled', value: { type: 'boolean', booleanValue: false }, isControl: true },
      ] : fg.type === 'string' ? [
        { id: 'var-v1', name: 'Algorithm V1', value: { type: 'string', stringValue: 'v1' }, isControl: true },
        { id: 'var-v2', name: 'Algorithm V2', value: { type: 'string', stringValue: 'v2' }, isControl: false },
        { id: 'var-v3', name: 'Algorithm V3', value: { type: 'string', stringValue: 'v3' }, isControl: false },
      ] : fg.type === 'number' ? [
        { id: 'var-20', name: '20 Items', value: { type: 'number', numberValue: 20 }, isControl: true },
        { id: 'var-50', name: '50 Items', value: { type: 'number', numberValue: 50 }, isControl: false },
        { id: 'var-100', name: '100 Items', value: { type: 'number', numberValue: 100 }, isControl: false },
      ] : fg.type === 'multivariate' ? [
        { id: 'var-a', name: 'Plan A', value: { type: 'json', jsonValue: { price: 9.99, features: ['basic'] } }, weight: 33, isControl: true },
        { id: 'var-b', name: 'Plan B', value: { type: 'json', jsonValue: { price: 14.99, features: ['basic', 'pro'] } }, weight: 33, isControl: false },
        { id: 'var-c', name: 'Plan C', value: { type: 'json', jsonValue: { price: 19.99, features: ['basic', 'pro', 'enterprise'] } }, weight: 34, isControl: false },
      ] : [
        { id: 'var-default', name: 'Default', value: { type: 'json', jsonValue: { features: [] } }, isControl: true },
        { id: 'var-full', name: 'Full Features', value: { type: 'json', jsonValue: { features: ['feature1', 'feature2', 'feature3'] } }, isControl: false },
      ];

      const flag: FeatureFlag = {
        id: flagId,
        key: fg.key,
        name: fg.name,
        description: `${fg.name} feature flag`,
        type: fg.type,
        project: 'proj-0001',
        environment: 'production',
        defaultValue: variations[0].value,
        variations,
        targeting: {
          enabled: fg.status === 'active',
          rules: idx < 3 ? [
            { id: 'rule-1', name: 'Beta Users', priority: 1, conditions: [{ attribute: 'segment', operator: 'in', value: 'beta-users' }], variation: 'var-true', enabled: true },
          ] : [],
          defaultVariation: variations[1]?.id || variations[0].id,
          offVariation: variations.find((v) => v.isControl)?.id,
          userTargeting: { enabled: idx % 2 === 0, includedUsers: idx === 0 ? ['user-vip'] : [], excludedUsers: [], userGroups: [] },
          contextTargeting: { enabled: false, contexts: [] },
        },
        scheduling: {
          enabled: idx === 6,
          schedules: idx === 6 ? [{ id: 'sched-1', name: 'Enable Next Week', action: 'enable', startTime: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), executed: false }] : [],
          timezone: 'UTC',
        },
        rollout: {
          strategy: idx === 5 ? 'gradual' : 'percentage',
          percentage: fg.status === 'active' ? (idx < 3 ? 100 : 50 + idx * 10) : 0,
          stages: [],
          bucketBy: 'key',
          gradualRollout: idx === 5 ? { initialPercentage: 10, increment: 10, intervalMinutes: 60, targetPercentage: 100, currentPercentage: 50, pauseOnError: true } : undefined,
        },
        prerequisites: idx === 2 ? [{ flagKey: 'beta-features', variation: 'var-full' }] : [],
        metrics: {
          enabled: idx === 5,
          goals: idx === 5 ? [
            { id: 'goal-1', name: 'Conversion Rate', type: 'conversion', event: 'purchase', successCriteria: { minEffect: 5, minConfidence: 95, direction: 'increase' } },
            { id: 'goal-2', name: 'Revenue', type: 'numeric', event: 'revenue', target: 1000 },
          ] : [],
          experiments: idx === 5 ? [{
            id: 'exp-1',
            name: 'Pricing Test',
            hypothesis: 'Plan B will increase conversions by 10%',
            status: 'running',
            variations: [{ variationId: 'var-a', name: 'Plan A', weight: 33, isControl: true }, { variationId: 'var-b', name: 'Plan B', weight: 33, isControl: false }, { variationId: 'var-c', name: 'Plan C', weight: 34, isControl: false }],
            traffic: { allocation: 100, samplingRate: 1, bucketBy: 'key' },
            metrics: [{ goalId: 'goal-1', name: 'Conversion Rate', isPrimary: true, minimumSampleSize: 1000, minimumDetectableEffect: 5 }],
            results: { startDate: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000), totalParticipants: 5000, variationResults: [{ variationId: 'var-a', participants: 1650, conversions: 165, conversionRate: 10, isWinner: false }, { variationId: 'var-b', participants: 1650, conversions: 198, conversionRate: 12, improvement: 20, confidence: 92, isWinner: true }, { variationId: 'var-c', participants: 1700, conversions: 170, conversionRate: 10, improvement: 0, confidence: 50, isWinner: false }], significance: 92, confidence: 92 },
            scheduling: { minDuration: 14, autoStop: true, stopCriteria: { minSampleSize: 1000, minSignificance: 95, maxPValue: 0.05 } },
            metadata: { createdAt: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000), createdBy: 'product-manager', updatedAt: new Date(), tags: ['pricing'] },
          }] : [],
          currentExperiment: idx === 5 ? 'exp-1' : undefined,
        },
        audit: {
          enabled: true,
          events: [
            { id: 'audit-1', timestamp: new Date(), action: 'updated', actor: 'admin', actorType: 'user', details: { rollout: 50 } },
            { id: 'audit-2', timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), action: 'enabled', actor: 'admin', actorType: 'user', details: {} },
          ],
          retentionDays: 90,
        },
        ownership: { owner: 'product-team', team: 'product', maintainers: ['tech-lead'], reviewers: ['qa-lead'], stakeholders: ['pm'] },
        status: fg.status,
        metadata: {
          createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
          createdBy: 'admin',
          updatedAt: new Date(),
          updatedBy: 'admin',
          version: 5 + idx,
          tags: [fg.type, fg.status],
          labels: { team: 'product' },
          temporary: idx === 2,
          expirationDate: idx === 2 ? new Date(Date.now() + 90 * 24 * 60 * 60 * 1000) : undefined,
        },
      };
      this.flags.set(flagId, flag);
    });

    // Initialize Change Requests
    const changeRequest: FlagChangeRequest = {
      id: 'cr-0001',
      flagKey: 'new-dashboard',
      project: 'proj-0001',
      environment: 'production',
      requestor: { id: 'dev-1', name: 'Developer', email: 'dev@example.com', reason: 'Roll out to 100% of users' },
      changes: [{ field: 'rollout.percentage', previousValue: 50, newValue: 100 }],
      review: {
        required: true,
        reviewers: [{ id: 'reviewer-1', name: 'Tech Lead', email: 'lead@example.com', status: 'pending' }],
        approvals: [],
        comments: [],
        status: 'pending',
      },
      status: 'pending',
      metadata: { createdAt: new Date(), updatedAt: new Date() },
    };
    this.changeRequests.set(changeRequest.id, changeRequest);
  }

  // Flag Operations
  public getFlags(project?: string, status?: FlagStatus): FeatureFlag[] {
    let flags = Array.from(this.flags.values());
    if (project) flags = flags.filter((f) => f.project === project);
    if (status) flags = flags.filter((f) => f.status === status);
    return flags;
  }

  public getFlagById(id: string): FeatureFlag | undefined {
    return this.flags.get(id);
  }

  public getFlagByKey(key: string, project?: string): FeatureFlag | undefined {
    return Array.from(this.flags.values()).find(
      (f) => f.key === key && (!project || f.project === project)
    );
  }

  // Project Operations
  public getProjects(): FlagProject[] {
    return Array.from(this.projects.values());
  }

  public getProjectById(id: string): FlagProject | undefined {
    return this.projects.get(id);
  }

  // Segment Operations
  public getSegments(project?: string): UserSegment[] {
    let segments = Array.from(this.segments.values());
    if (project) segments = segments.filter((s) => s.project === project);
    return segments;
  }

  public getSegmentById(id: string): UserSegment | undefined {
    return this.segments.get(id);
  }

  // Change Request Operations
  public getChangeRequests(status?: FlagChangeRequest['status']): FlagChangeRequest[] {
    let requests = Array.from(this.changeRequests.values());
    if (status) requests = requests.filter((r) => r.status === status);
    return requests;
  }

  public getChangeRequestById(id: string): FlagChangeRequest | undefined {
    return this.changeRequests.get(id);
  }

  // Evaluation
  public evaluateFlag(flagKey: string, context: EvaluationContext): FlagEvaluation | undefined {
    const flag = this.getFlagByKey(flagKey);
    if (!flag) return undefined;

    const defaultVariation = flag.variations.find((v) => v.id === flag.targeting.defaultVariation) || flag.variations[0];
    return {
      flagKey,
      value: defaultVariation.value,
      variationIndex: flag.variations.indexOf(defaultVariation),
      variationId: defaultVariation.id,
      reason: { kind: flag.targeting.enabled ? 'FALLTHROUGH' : 'OFF' },
      context,
      timestamp: new Date(),
      trackEvents: true,
    };
  }

  // Statistics
  public getStatistics(): FeatureFlagStatistics {
    const flags = Array.from(this.flags.values());
    const experiments = flags.flatMap((f) => f.metrics.experiments || []);

    const byType: Record<FlagType, number> = { boolean: 0, string: 0, number: 0, json: 0, multivariate: 0 };
    const byStatus: Record<FlagStatus, number> = { active: 0, inactive: 0, archived: 0, scheduled: 0 };
    const byProject: Record<string, number> = {};
    const byEnvironment: Record<string, number> = {};

    flags.forEach((f) => {
      byType[f.type]++;
      byStatus[f.status]++;
      byProject[f.project] = (byProject[f.project] || 0) + 1;
      byEnvironment[f.environment] = (byEnvironment[f.environment] || 0) + 1;
    });

    return {
      overview: {
        totalFlags: flags.length,
        activeFlags: byStatus.active,
        inactiveFlags: byStatus.inactive,
        archivedFlags: byStatus.archived,
        temporaryFlags: flags.filter((f) => f.metadata.temporary).length,
        permanentFlags: flags.filter((f) => !f.metadata.temporary).length,
        expiringFlags: flags.filter((f) => f.metadata.expirationDate && new Date(f.metadata.expirationDate) <= new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)).length,
      },
      byType,
      byStatus,
      byProject,
      byEnvironment,
      experiments: {
        active: experiments.filter((e) => e.status === 'running').length,
        completed: experiments.filter((e) => e.status === 'completed').length,
        paused: experiments.filter((e) => e.status === 'paused').length,
        avgDuration: 14,
        winRate: 60,
      },
      evaluations: {
        total: Math.floor(Math.random() * 10000000),
        today: Math.floor(Math.random() * 100000),
        avgLatency: 5,
        errorRate: 0.01,
      },
      changes: {
        today: 5,
        thisWeek: 25,
        pendingApproval: this.changeRequests.size,
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

export const featureFlagService = FeatureFlagService.getInstance();
export type {
  FlagType,
  FlagStatus,
  RolloutStrategy,
  TargetingOperator,
  FeatureFlag,
  FlagValue,
  FlagVariation,
  FlagTargeting,
  TargetingRule,
  TargetingCondition,
  UserTargeting,
  UserGroup,
  ContextTargeting,
  ContextRule,
  FlagScheduling,
  FlagSchedule,
  ScheduleRecurrence,
  FlagRollout,
  RolloutStage,
  RolloutCriteria,
  MetricCriteria,
  GradualRollout,
  RingRollout,
  RolloutRing,
  FlagPrerequisite,
  FlagMetrics,
  FlagGoal,
  GoalCriteria,
  FlagExperiment,
  ExperimentVariation,
  ExperimentTraffic,
  ExperimentMetric,
  ExperimentResults,
  VariationResult,
  ExperimentScheduling,
  StopCriteria,
  ExperimentMetadata,
  FlagAudit,
  FlagAuditEvent,
  FlagOwnership,
  FlagMetadata,
  FlagProject,
  ProjectEnvironment,
  ProjectSettings,
  ApprovalSettings,
  ProjectAccess,
  AccessRole,
  ProjectIntegration,
  ProjectMetadata,
  UserSegment,
  SegmentRule,
  SegmentClause,
  SegmentMetadata,
  FlagEvaluation,
  EvaluationReason,
  EvaluationContext,
  FlagChangeRequest,
  ChangeRequestor,
  FlagChange,
  ChangeImpact,
  ChangeReview,
  ReviewerInfo,
  ReviewApproval,
  ReviewComment,
  ChangeScheduling,
  ChangeRequestMetadata,
  FeatureFlagStatistics,
};
