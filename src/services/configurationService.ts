/**
 * Configuration Service
 * Comprehensive configuration management, versioning, and distribution
 */

// Configuration Type
type ConfigurationType = 'application' | 'environment' | 'feature' | 'secret' | 'runtime' | 'static';

// Configuration Format
type ConfigurationFormat = 'json' | 'yaml' | 'toml' | 'properties' | 'env' | 'xml' | 'hcl';

// Configuration Status
type ConfigurationStatus = 'draft' | 'pending' | 'active' | 'deprecated' | 'archived';

// Change Type
type ChangeType = 'create' | 'update' | 'delete' | 'rollback' | 'promote';

// Configuration
interface Configuration {
  id: string;
  name: string;
  description: string;
  type: ConfigurationType;
  format: ConfigurationFormat;
  namespace: string;
  environment: string;
  version: ConfigurationVersion;
  content: ConfigurationContent;
  schema: ConfigurationSchema;
  inheritance: ConfigurationInheritance;
  deployment: ConfigurationDeployment;
  validation: ConfigurationValidation;
  audit: ConfigurationAudit;
  access: ConfigurationAccess;
  status: ConfigurationStatus;
  metadata: ConfigurationMetadata;
}

// Configuration Version
interface ConfigurationVersion {
  current: string;
  previous?: string;
  history: VersionHistoryEntry[];
  latest: string;
  publishedAt?: Date;
  publishedBy?: string;
}

// Version History Entry
interface VersionHistoryEntry {
  version: string;
  timestamp: Date;
  author: string;
  message: string;
  changeType: ChangeType;
  diff?: ConfigurationDiff;
  approved: boolean;
  approver?: string;
}

// Configuration Diff
interface ConfigurationDiff {
  added: string[];
  removed: string[];
  modified: DiffEntry[];
  summary: string;
}

// Diff Entry
interface DiffEntry {
  path: string;
  oldValue: unknown;
  newValue: unknown;
}

// Configuration Content
interface ConfigurationContent {
  data: Record<string, unknown>;
  raw?: string;
  encrypted: boolean;
  encryptedFields?: string[];
  resolvedData?: Record<string, unknown>;
  references: ConfigurationReference[];
}

// Configuration Reference
interface ConfigurationReference {
  path: string;
  type: 'config' | 'secret' | 'environment' | 'external';
  source: string;
  resolved: boolean;
  value?: unknown;
}

// Configuration Schema
interface ConfigurationSchema {
  enabled: boolean;
  type: 'json_schema' | 'avro' | 'protobuf' | 'custom';
  definition?: Record<string, unknown>;
  version?: string;
  validation: SchemaValidation;
}

// Schema Validation
interface SchemaValidation {
  strict: boolean;
  additionalProperties: boolean;
  coerceTypes: boolean;
  removeAdditional: boolean;
  useDefaults: boolean;
}

// Configuration Inheritance
interface ConfigurationInheritance {
  enabled: boolean;
  parent?: string;
  mergeStrategy: 'override' | 'merge' | 'deep_merge';
  inheritedFields: string[];
  excludedFields: string[];
  overrides: Record<string, unknown>;
}

// Configuration Deployment
interface ConfigurationDeployment {
  targets: DeploymentTarget[];
  strategy: DeploymentStrategy;
  rollout: RolloutConfig;
  rollback: RollbackConfig;
  notifications: DeploymentNotification[];
  lastDeployment?: DeploymentRecord;
}

// Deployment Target
interface DeploymentTarget {
  id: string;
  name: string;
  type: 'application' | 'service' | 'cluster' | 'region' | 'instance';
  selector: TargetSelector;
  status: 'pending' | 'synced' | 'failed' | 'outdated';
  lastSync?: Date;
  version?: string;
}

// Target Selector
interface TargetSelector {
  labels?: Record<string, string>;
  namespaces?: string[];
  environments?: string[];
  applications?: string[];
}

// Deployment Strategy
interface DeploymentStrategy {
  type: 'immediate' | 'rolling' | 'canary' | 'blue_green';
  batchSize?: number;
  batchPercentage?: number;
  interval?: number;
  pauseOnFailure: boolean;
  autoPromote: boolean;
}

// Rollout Config
interface RolloutConfig {
  enabled: boolean;
  initialPercentage: number;
  increment: number;
  interval: number;
  healthCheck: HealthCheckConfig;
  metrics: RolloutMetrics;
}

// Health Check Config
interface HealthCheckConfig {
  enabled: boolean;
  endpoint?: string;
  interval: number;
  timeout: number;
  successThreshold: number;
  failureThreshold: number;
}

// Rollout Metrics
interface RolloutMetrics {
  errorRateThreshold: number;
  latencyThreshold: number;
  successRateThreshold: number;
  evaluationInterval: number;
}

// Rollback Config
interface RollbackConfig {
  enabled: boolean;
  automatic: boolean;
  threshold: number;
  window: number;
  keepVersions: number;
  notifyOnRollback: boolean;
}

// Deployment Notification
interface DeploymentNotification {
  channel: 'email' | 'slack' | 'webhook' | 'pagerduty';
  events: ('started' | 'completed' | 'failed' | 'rolled_back')[];
  recipients: string[];
  config: Record<string, string>;
}

// Deployment Record
interface DeploymentRecord {
  id: string;
  version: string;
  timestamp: Date;
  status: 'success' | 'failed' | 'in_progress' | 'rolled_back';
  targets: number;
  successfulTargets: number;
  failedTargets: number;
  duration: number;
  initiatedBy: string;
  error?: string;
}

// Configuration Validation
interface ConfigurationValidation {
  enabled: boolean;
  rules: ValidationRule[];
  lastValidation?: ValidationResult;
  validateOnChange: boolean;
  validateOnDeploy: boolean;
}

// Validation Rule
interface ValidationRule {
  id: string;
  name: string;
  description: string;
  type: 'schema' | 'regex' | 'range' | 'enum' | 'custom';
  path: string;
  condition: ValidationCondition;
  severity: 'error' | 'warning' | 'info';
  message: string;
  enabled: boolean;
}

// Validation Condition
interface ValidationCondition {
  operator: 'equals' | 'not_equals' | 'contains' | 'matches' | 'in' | 'not_in' | 'gt' | 'lt' | 'gte' | 'lte';
  value?: unknown;
  values?: unknown[];
  pattern?: string;
  min?: number;
  max?: number;
}

// Validation Result
interface ValidationResult {
  valid: boolean;
  timestamp: Date;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  duration: number;
}

// Validation Error
interface ValidationError {
  rule: string;
  path: string;
  message: string;
  value: unknown;
}

// Validation Warning
interface ValidationWarning {
  rule: string;
  path: string;
  message: string;
  suggestion?: string;
}

// Configuration Audit
interface ConfigurationAudit {
  enabled: boolean;
  events: AuditEvent[];
  retentionDays: number;
  exportEnabled: boolean;
}

// Audit Event
interface AuditEvent {
  id: string;
  timestamp: Date;
  action: 'view' | 'create' | 'update' | 'delete' | 'deploy' | 'rollback' | 'export';
  actor: string;
  actorType: 'user' | 'service' | 'system';
  details: Record<string, unknown>;
  ipAddress?: string;
  result: 'success' | 'failure';
  version?: string;
}

// Configuration Access
interface ConfigurationAccess {
  owner: string;
  team: string;
  visibility: 'public' | 'internal' | 'private' | 'restricted';
  permissions: AccessPermission[];
  apiKeys: ConfigApiKey[];
  tokens: AccessToken[];
}

// Access Permission
interface AccessPermission {
  principal: string;
  principalType: 'user' | 'group' | 'service' | 'role';
  actions: ('read' | 'write' | 'delete' | 'deploy' | 'admin')[];
  conditions?: AccessCondition[];
}

// Access Condition
interface AccessCondition {
  type: 'ip' | 'time' | 'environment' | 'mfa';
  value: string;
}

// Config API Key
interface ConfigApiKey {
  id: string;
  name: string;
  key: string;
  permissions: string[];
  rateLimit?: number;
  expiresAt?: Date;
  lastUsed?: Date;
  status: 'active' | 'revoked';
}

// Access Token
interface AccessToken {
  id: string;
  name: string;
  token: string;
  scopes: string[];
  expiresAt: Date;
  createdBy: string;
  status: 'active' | 'expired' | 'revoked';
}

// Configuration Metadata
interface ConfigurationMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  updatedBy: string;
  tags: string[];
  labels: Record<string, string>;
  annotations: Record<string, string>;
  externalId?: string;
}

// Configuration Template
interface ConfigurationTemplate {
  id: string;
  name: string;
  description: string;
  type: ConfigurationType;
  format: ConfigurationFormat;
  schema: ConfigurationSchema;
  defaults: Record<string, unknown>;
  variables: TemplateVariable[];
  examples: TemplateExample[];
  usage: TemplateUsage;
  metadata: TemplateMetadata;
}

// Template Variable
interface TemplateVariable {
  name: string;
  description: string;
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  required: boolean;
  default?: unknown;
  validation?: ValidationCondition;
  sensitive: boolean;
}

// Template Example
interface TemplateExample {
  name: string;
  description: string;
  values: Record<string, unknown>;
  result: Record<string, unknown>;
}

// Template Usage
interface TemplateUsage {
  totalUsage: number;
  lastUsed?: Date;
  configurations: string[];
}

// Template Metadata
interface TemplateMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  version: string;
  category: string;
  tags: string[];
}

// Configuration Environment
interface ConfigurationEnvironment {
  id: string;
  name: string;
  description: string;
  type: 'development' | 'staging' | 'production' | 'testing';
  variables: EnvironmentVariable[];
  secrets: EnvironmentSecret[];
  configurations: string[];
  promotion: PromotionConfig;
  protection: EnvironmentProtection;
  metadata: EnvironmentMetadata;
}

// Environment Variable
interface EnvironmentVariable {
  name: string;
  value: string;
  description?: string;
  type: 'plain' | 'reference' | 'computed';
  source?: string;
  sensitive: boolean;
  overridable: boolean;
}

// Environment Secret
interface EnvironmentSecret {
  name: string;
  reference: string;
  version?: string;
  provider: 'vault' | 'aws_secrets' | 'azure_keyvault' | 'gcp_secrets' | 'local';
  rotationEnabled: boolean;
  lastRotated?: Date;
}

// Promotion Config
interface PromotionConfig {
  enabled: boolean;
  sourceEnvironments: string[];
  targetEnvironments: string[];
  approvalRequired: boolean;
  approvers: string[];
  autoPromote: boolean;
  promotionWindow?: {
    days: number[];
    startTime: string;
    endTime: string;
    timezone: string;
  };
}

// Environment Protection
interface EnvironmentProtection {
  enabled: boolean;
  rules: ProtectionRule[];
  breakGlass: BreakGlassConfig;
}

// Protection Rule
interface ProtectionRule {
  id: string;
  name: string;
  type: 'approval' | 'review' | 'wait_timer' | 'branch_protection';
  config: Record<string, unknown>;
  enforced: boolean;
}

// Break Glass Config
interface BreakGlassConfig {
  enabled: boolean;
  approvers: string[];
  expirationMinutes: number;
  auditRequired: boolean;
}

// Environment Metadata
interface EnvironmentMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  region?: string;
  cluster?: string;
}

// Configuration Namespace
interface ConfigurationNamespace {
  id: string;
  name: string;
  description: string;
  parent?: string;
  children: string[];
  configurations: string[];
  quotas: NamespaceQuota;
  policies: NamespacePolicy[];
  access: NamespaceAccess;
  metadata: NamespaceMetadata;
}

// Namespace Quota
interface NamespaceQuota {
  maxConfigurations: number;
  maxVersions: number;
  maxSize: number;
  usedConfigurations: number;
  usedVersions: number;
  usedSize: number;
}

// Namespace Policy
interface NamespacePolicy {
  id: string;
  name: string;
  type: 'naming' | 'validation' | 'deployment' | 'access';
  rules: Record<string, unknown>;
  enforced: boolean;
}

// Namespace Access
interface NamespaceAccess {
  owner: string;
  admins: string[];
  members: string[];
  viewers: string[];
}

// Namespace Metadata
interface NamespaceMetadata {
  createdAt: Date;
  createdBy: string;
  updatedAt: Date;
  labels: Record<string, string>;
}

// Configuration Change Request
interface ConfigurationChangeRequest {
  id: string;
  configurationId: string;
  type: ChangeType;
  status: 'pending' | 'approved' | 'rejected' | 'applied' | 'cancelled';
  requestor: ChangeRequestor;
  changes: ProposedChange[];
  review: ChangeReview;
  impact: ChangeImpact;
  schedule?: ChangeSchedule;
  metadata: ChangeRequestMetadata;
}

// Change Requestor
interface ChangeRequestor {
  id: string;
  name: string;
  email: string;
  team: string;
  justification: string;
}

// Proposed Change
interface ProposedChange {
  path: string;
  operation: 'add' | 'remove' | 'replace';
  oldValue?: unknown;
  newValue?: unknown;
}

// Change Review
interface ChangeReview {
  required: boolean;
  reviewers: Reviewer[];
  approvals: ReviewApproval[];
  status: 'pending' | 'approved' | 'rejected';
  comments: ReviewComment[];
}

// Reviewer
interface Reviewer {
  id: string;
  name: string;
  email: string;
  role: 'required' | 'optional';
  status: 'pending' | 'approved' | 'rejected' | 'abstained';
}

// Review Approval
interface ReviewApproval {
  reviewerId: string;
  timestamp: Date;
  status: 'approved' | 'rejected' | 'abstained';
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

// Change Impact
interface ChangeImpact {
  affectedServices: string[];
  affectedEnvironments: string[];
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  impactAnalysis: string;
  rollbackPlan: string;
}

// Change Schedule
interface ChangeSchedule {
  scheduledAt: Date;
  timezone: string;
  maintenanceWindow: boolean;
  autoApply: boolean;
}

// Change Request Metadata
interface ChangeRequestMetadata {
  createdAt: Date;
  updatedAt: Date;
  appliedAt?: Date;
  ticketId?: string;
}

// Configuration Statistics
interface ConfigurationStatistics {
  overview: {
    totalConfigurations: number;
    activeConfigurations: number;
    draftConfigurations: number;
    deprecatedConfigurations: number;
    totalVersions: number;
    totalNamespaces: number;
    totalEnvironments: number;
  };
  byType: Record<ConfigurationType, number>;
  byFormat: Record<ConfigurationFormat, number>;
  byStatus: Record<ConfigurationStatus, number>;
  byEnvironment: Record<string, number>;
  changes: {
    today: number;
    thisWeek: number;
    thisMonth: number;
    pendingApproval: number;
  };
  deployments: {
    successful: number;
    failed: number;
    rolledBack: number;
    avgDeploymentTime: number;
  };
  validation: {
    validConfigurations: number;
    invalidConfigurations: number;
    totalValidations: number;
    avgValidationTime: number;
  };
}

class ConfigurationService {
  private static instance: ConfigurationService;
  private configurations: Map<string, Configuration> = new Map();
  private templates: Map<string, ConfigurationTemplate> = new Map();
  private environments: Map<string, ConfigurationEnvironment> = new Map();
  private namespaces: Map<string, ConfigurationNamespace> = new Map();
  private changeRequests: Map<string, ConfigurationChangeRequest> = new Map();
  private eventListeners: ((event: string, data: unknown) => void)[] = [];

  private constructor() {
    this.initializeSampleData();
  }

  public static getInstance(): ConfigurationService {
    if (!ConfigurationService.instance) {
      ConfigurationService.instance = new ConfigurationService();
    }
    return ConfigurationService.instance;
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private initializeSampleData(): void {
    // Initialize Namespaces
    const namespacesData = ['default', 'platform', 'commerce', 'security'];
    namespacesData.forEach((ns, idx) => {
      const namespace: ConfigurationNamespace = {
        id: `ns-${(idx + 1).toString().padStart(4, '0')}`,
        name: ns,
        description: `${ns} namespace for configurations`,
        configurations: [],
        children: [],
        quotas: { maxConfigurations: 100, maxVersions: 50, maxSize: 10485760, usedConfigurations: 0, usedVersions: 0, usedSize: 0 },
        policies: [],
        access: { owner: 'admin', admins: ['admin'], members: [`${ns}-team`], viewers: [] },
        metadata: { createdAt: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), createdBy: 'admin', updatedAt: new Date(), labels: {} },
      };
      this.namespaces.set(namespace.id, namespace);
    });

    // Initialize Environments
    const envsData = [
      { name: 'development', type: 'development' as const },
      { name: 'staging', type: 'staging' as const },
      { name: 'production', type: 'production' as const },
    ];
    envsData.forEach((env, idx) => {
      const environment: ConfigurationEnvironment = {
        id: `env-${(idx + 1).toString().padStart(4, '0')}`,
        name: env.name,
        description: `${env.name} environment`,
        type: env.type,
        variables: [
          { name: 'LOG_LEVEL', value: env.type === 'production' ? 'info' : 'debug', type: 'plain', sensitive: false, overridable: true },
          { name: 'API_URL', value: `https://api-${env.name}.example.com`, type: 'plain', sensitive: false, overridable: false },
          { name: 'DB_HOST', value: `db-${env.name}.example.com`, type: 'plain', sensitive: false, overridable: false },
        ],
        secrets: [
          { name: 'DB_PASSWORD', reference: `vault://secrets/${env.name}/db-password`, provider: 'vault', rotationEnabled: true, lastRotated: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) },
          { name: 'API_KEY', reference: `vault://secrets/${env.name}/api-key`, provider: 'vault', rotationEnabled: true },
        ],
        configurations: [],
        promotion: {
          enabled: true,
          sourceEnvironments: idx > 0 ? [envsData[idx - 1].name] : [],
          targetEnvironments: idx < envsData.length - 1 ? [envsData[idx + 1].name] : [],
          approvalRequired: env.type === 'production',
          approvers: env.type === 'production' ? ['tech-lead', 'ops-lead'] : [],
          autoPromote: env.type !== 'production',
        },
        protection: {
          enabled: env.type === 'production',
          rules: env.type === 'production' ? [
            { id: 'rule-1', name: 'Approval Required', type: 'approval', config: { minApprovals: 2 }, enforced: true },
            { id: 'rule-2', name: 'Maintenance Window', type: 'wait_timer', config: { hours: 'business' }, enforced: false },
          ] : [],
          breakGlass: { enabled: env.type === 'production', approvers: ['admin'], expirationMinutes: 60, auditRequired: true },
        },
        metadata: { createdAt: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), createdBy: 'admin', updatedAt: new Date(), region: 'us-west-2' },
      };
      this.environments.set(environment.id, environment);
    });

    // Initialize Configurations
    const configsData = [
      { name: 'app-config', type: 'application' as ConfigurationType, namespace: 'default' },
      { name: 'database-config', type: 'application' as ConfigurationType, namespace: 'platform' },
      { name: 'cache-config', type: 'application' as ConfigurationType, namespace: 'platform' },
      { name: 'security-config', type: 'application' as ConfigurationType, namespace: 'security' },
      { name: 'feature-flags', type: 'feature' as ConfigurationType, namespace: 'default' },
      { name: 'rate-limits', type: 'runtime' as ConfigurationType, namespace: 'platform' },
      { name: 'logging-config', type: 'application' as ConfigurationType, namespace: 'platform' },
      { name: 'notification-config', type: 'application' as ConfigurationType, namespace: 'commerce' },
    ];

    configsData.forEach((cfg, idx) => {
      const configId = `cfg-${(idx + 1).toString().padStart(4, '0')}`;
      const config: Configuration = {
        id: configId,
        name: cfg.name,
        description: `${cfg.name} configuration`,
        type: cfg.type,
        format: 'json',
        namespace: cfg.namespace,
        environment: 'production',
        version: {
          current: '1.5.0',
          previous: '1.4.0',
          history: [
            { version: '1.5.0', timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), author: 'admin', message: 'Updated settings', changeType: 'update', approved: true, approver: 'tech-lead' },
            { version: '1.4.0', timestamp: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), author: 'developer', message: 'Added new features', changeType: 'update', approved: true, approver: 'admin' },
            { version: '1.0.0', timestamp: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000), author: 'admin', message: 'Initial configuration', changeType: 'create', approved: true },
          ],
          latest: '1.5.0',
          publishedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          publishedBy: 'admin',
        },
        content: {
          data: cfg.type === 'feature' ? {
            features: {
              newDashboard: { enabled: true, rolloutPercentage: 100 },
              betaFeature: { enabled: false, allowedUsers: ['beta-testers'] },
              experimentalApi: { enabled: true, rolloutPercentage: 25 },
            },
          } : {
            server: { port: 8080, host: '0.0.0.0', timeout: 30000 },
            database: { pool: { min: 5, max: 20 }, timeout: 5000 },
            cache: { enabled: true, ttl: 3600 },
            logging: { level: 'info', format: 'json' },
          },
          encrypted: false,
          references: [],
        },
        schema: {
          enabled: true,
          type: 'json_schema',
          validation: { strict: true, additionalProperties: false, coerceTypes: false, removeAdditional: false, useDefaults: true },
        },
        inheritance: {
          enabled: idx % 2 === 0,
          mergeStrategy: 'deep_merge',
          inheritedFields: [],
          excludedFields: [],
          overrides: {},
        },
        deployment: {
          targets: [
            { id: `target-${idx}-1`, name: 'Production Cluster', type: 'cluster', selector: { environments: ['production'] }, status: 'synced', lastSync: new Date(), version: '1.5.0' },
            { id: `target-${idx}-2`, name: 'Staging Cluster', type: 'cluster', selector: { environments: ['staging'] }, status: 'synced', lastSync: new Date(), version: '1.5.0' },
          ],
          strategy: { type: 'rolling', batchPercentage: 25, interval: 30, pauseOnFailure: true, autoPromote: false },
          rollout: { enabled: true, initialPercentage: 10, increment: 20, interval: 300, healthCheck: { enabled: true, interval: 30, timeout: 10, successThreshold: 2, failureThreshold: 3 }, metrics: { errorRateThreshold: 5, latencyThreshold: 1000, successRateThreshold: 99, evaluationInterval: 60 } },
          rollback: { enabled: true, automatic: true, threshold: 10, window: 300, keepVersions: 10, notifyOnRollback: true },
          notifications: [{ channel: 'slack', events: ['started', 'completed', 'failed', 'rolled_back'], recipients: ['#deployments'], config: {} }],
          lastDeployment: { id: 'deploy-1', version: '1.5.0', timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), status: 'success', targets: 2, successfulTargets: 2, failedTargets: 0, duration: 120, initiatedBy: 'admin' },
        },
        validation: {
          enabled: true,
          rules: [
            { id: 'rule-1', name: 'Port Range', description: 'Port must be valid', type: 'range', path: 'server.port', condition: { operator: 'gte', min: 1, max: 65535 }, severity: 'error', message: 'Invalid port number', enabled: true },
            { id: 'rule-2', name: 'Timeout Positive', description: 'Timeout must be positive', type: 'range', path: 'server.timeout', condition: { operator: 'gt', value: 0 }, severity: 'error', message: 'Timeout must be positive', enabled: true },
          ],
          lastValidation: { valid: true, timestamp: new Date(), errors: [], warnings: [], duration: 15 },
          validateOnChange: true,
          validateOnDeploy: true,
        },
        audit: {
          enabled: true,
          events: [
            { id: 'audit-1', timestamp: new Date(), action: 'view', actor: 'admin', actorType: 'user', details: {}, result: 'success', version: '1.5.0' },
            { id: 'audit-2', timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), action: 'deploy', actor: 'admin', actorType: 'user', details: { targets: 2 }, result: 'success', version: '1.5.0' },
          ],
          retentionDays: 365,
          exportEnabled: true,
        },
        access: {
          owner: 'platform-team',
          team: 'platform',
          visibility: 'internal',
          permissions: [
            { principal: 'platform-team', principalType: 'group', actions: ['read', 'write', 'deploy', 'admin'] },
            { principal: 'developers', principalType: 'group', actions: ['read'] },
          ],
          apiKeys: [],
          tokens: [],
        },
        status: 'active',
        metadata: {
          createdAt: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000),
          createdBy: 'admin',
          updatedAt: new Date(),
          updatedBy: 'admin',
          tags: [cfg.type, cfg.namespace],
          labels: { type: cfg.type, namespace: cfg.namespace },
          annotations: {},
        },
      };
      this.configurations.set(configId, config);
    });

    // Initialize Templates
    const templatesData = [
      { name: 'Application Config', type: 'application' as ConfigurationType },
      { name: 'Database Config', type: 'application' as ConfigurationType },
      { name: 'Feature Flags', type: 'feature' as ConfigurationType },
    ];

    templatesData.forEach((tmpl, idx) => {
      const template: ConfigurationTemplate = {
        id: `tmpl-${(idx + 1).toString().padStart(4, '0')}`,
        name: tmpl.name,
        description: `Template for ${tmpl.name.toLowerCase()}`,
        type: tmpl.type,
        format: 'json',
        schema: { enabled: true, type: 'json_schema', validation: { strict: true, additionalProperties: false, coerceTypes: false, removeAdditional: false, useDefaults: true } },
        defaults: {},
        variables: [
          { name: 'environment', description: 'Target environment', type: 'string', required: true, sensitive: false },
          { name: 'region', description: 'Deployment region', type: 'string', required: false, default: 'us-west-2', sensitive: false },
        ],
        examples: [{ name: 'Basic Example', description: 'Basic configuration', values: { environment: 'production' }, result: {} }],
        usage: { totalUsage: Math.floor(Math.random() * 100), lastUsed: new Date(), configurations: [] },
        metadata: { createdAt: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), createdBy: 'admin', updatedAt: new Date(), version: '1.0.0', category: tmpl.type, tags: [tmpl.type] },
      };
      this.templates.set(template.id, template);
    });

    // Initialize Change Requests
    const changeRequest: ConfigurationChangeRequest = {
      id: 'cr-0001',
      configurationId: 'cfg-0001',
      type: 'update',
      status: 'pending',
      requestor: { id: 'dev-1', name: 'Developer', email: 'dev@example.com', team: 'platform', justification: 'Update timeout settings for better performance' },
      changes: [{ path: 'server.timeout', operation: 'replace', oldValue: 30000, newValue: 60000 }],
      review: {
        required: true,
        reviewers: [{ id: 'reviewer-1', name: 'Tech Lead', email: 'lead@example.com', role: 'required', status: 'pending' }],
        approvals: [],
        status: 'pending',
        comments: [],
      },
      impact: { affectedServices: ['api-service', 'web-service'], affectedEnvironments: ['staging', 'production'], riskLevel: 'low', impactAnalysis: 'Minor timeout change', rollbackPlan: 'Revert to previous version' },
      metadata: { createdAt: new Date(), updatedAt: new Date() },
    };
    this.changeRequests.set(changeRequest.id, changeRequest);
  }

  // Configuration Operations
  public getConfigurations(namespace?: string, type?: ConfigurationType): Configuration[] {
    let configs = Array.from(this.configurations.values());
    if (namespace) configs = configs.filter((c) => c.namespace === namespace);
    if (type) configs = configs.filter((c) => c.type === type);
    return configs;
  }

  public getConfigurationById(id: string): Configuration | undefined {
    return this.configurations.get(id);
  }

  public getConfigurationByName(name: string, namespace?: string): Configuration | undefined {
    return Array.from(this.configurations.values()).find(
      (c) => c.name === name && (!namespace || c.namespace === namespace)
    );
  }

  // Template Operations
  public getTemplates(type?: ConfigurationType): ConfigurationTemplate[] {
    let templates = Array.from(this.templates.values());
    if (type) templates = templates.filter((t) => t.type === type);
    return templates;
  }

  public getTemplateById(id: string): ConfigurationTemplate | undefined {
    return this.templates.get(id);
  }

  // Environment Operations
  public getEnvironments(): ConfigurationEnvironment[] {
    return Array.from(this.environments.values());
  }

  public getEnvironmentById(id: string): ConfigurationEnvironment | undefined {
    return this.environments.get(id);
  }

  // Namespace Operations
  public getNamespaces(): ConfigurationNamespace[] {
    return Array.from(this.namespaces.values());
  }

  public getNamespaceById(id: string): ConfigurationNamespace | undefined {
    return this.namespaces.get(id);
  }

  // Change Request Operations
  public getChangeRequests(status?: ConfigurationChangeRequest['status']): ConfigurationChangeRequest[] {
    let requests = Array.from(this.changeRequests.values());
    if (status) requests = requests.filter((r) => r.status === status);
    return requests;
  }

  public getChangeRequestById(id: string): ConfigurationChangeRequest | undefined {
    return this.changeRequests.get(id);
  }

  // Statistics
  public getStatistics(): ConfigurationStatistics {
    const configs = Array.from(this.configurations.values());
    const requests = Array.from(this.changeRequests.values());

    const byType: Record<ConfigurationType, number> = {
      application: 0, environment: 0, feature: 0, secret: 0, runtime: 0, static: 0,
    };
    const byFormat: Record<ConfigurationFormat, number> = {
      json: 0, yaml: 0, toml: 0, properties: 0, env: 0, xml: 0, hcl: 0,
    };
    const byStatus: Record<ConfigurationStatus, number> = {
      draft: 0, pending: 0, active: 0, deprecated: 0, archived: 0,
    };
    const byEnvironment: Record<string, number> = {};

    configs.forEach((c) => {
      byType[c.type]++;
      byFormat[c.format]++;
      byStatus[c.status]++;
      byEnvironment[c.environment] = (byEnvironment[c.environment] || 0) + 1;
    });

    return {
      overview: {
        totalConfigurations: configs.length,
        activeConfigurations: byStatus.active,
        draftConfigurations: byStatus.draft,
        deprecatedConfigurations: byStatus.deprecated,
        totalVersions: configs.reduce((sum, c) => sum + c.version.history.length, 0),
        totalNamespaces: this.namespaces.size,
        totalEnvironments: this.environments.size,
      },
      byType,
      byFormat,
      byStatus,
      byEnvironment,
      changes: {
        today: 5,
        thisWeek: 25,
        thisMonth: 100,
        pendingApproval: requests.filter((r) => r.status === 'pending').length,
      },
      deployments: {
        successful: configs.filter((c) => c.deployment.lastDeployment?.status === 'success').length,
        failed: configs.filter((c) => c.deployment.lastDeployment?.status === 'failed').length,
        rolledBack: configs.filter((c) => c.deployment.lastDeployment?.status === 'rolled_back').length,
        avgDeploymentTime: 120,
      },
      validation: {
        validConfigurations: configs.filter((c) => c.validation.lastValidation?.valid).length,
        invalidConfigurations: configs.filter((c) => !c.validation.lastValidation?.valid).length,
        totalValidations: configs.length * 5,
        avgValidationTime: 15,
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

export const configurationService = ConfigurationService.getInstance();
export type {
  ConfigurationType,
  ConfigurationFormat,
  ConfigurationStatus,
  ChangeType,
  Configuration,
  ConfigurationVersion,
  VersionHistoryEntry,
  ConfigurationDiff,
  DiffEntry,
  ConfigurationContent,
  ConfigurationReference,
  ConfigurationSchema,
  SchemaValidation,
  ConfigurationInheritance,
  ConfigurationDeployment,
  DeploymentTarget,
  TargetSelector,
  DeploymentStrategy,
  RolloutConfig,
  HealthCheckConfig,
  RolloutMetrics,
  RollbackConfig,
  DeploymentNotification,
  DeploymentRecord,
  ConfigurationValidation,
  ValidationRule,
  ValidationCondition,
  ValidationResult,
  ValidationError,
  ValidationWarning,
  ConfigurationAudit,
  AuditEvent,
  ConfigurationAccess,
  AccessPermission,
  AccessCondition,
  ConfigApiKey,
  AccessToken,
  ConfigurationMetadata,
  ConfigurationTemplate,
  TemplateVariable,
  TemplateExample,
  TemplateUsage,
  TemplateMetadata,
  ConfigurationEnvironment,
  EnvironmentVariable,
  EnvironmentSecret,
  PromotionConfig,
  EnvironmentProtection,
  ProtectionRule,
  BreakGlassConfig,
  EnvironmentMetadata,
  ConfigurationNamespace,
  NamespaceQuota,
  NamespacePolicy,
  NamespaceAccess,
  NamespaceMetadata,
  ConfigurationChangeRequest,
  ChangeRequestor,
  ProposedChange,
  ChangeReview,
  Reviewer,
  ReviewApproval,
  ReviewComment,
  ChangeImpact,
  ChangeSchedule,
  ChangeRequestMetadata,
  ConfigurationStatistics,
};
