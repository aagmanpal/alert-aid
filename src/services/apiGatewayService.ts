/**
 * API Gateway Service
 * API management, routing, rate limiting, authentication, and request/response transformation
 */

// HTTP method
type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'HEAD' | 'OPTIONS';

// API status
type ApiStatus = 'active' | 'inactive' | 'deprecated' | 'beta' | 'maintenance';

// Auth type
type AuthType = 'none' | 'api_key' | 'bearer' | 'basic' | 'oauth2' | 'jwt' | 'hmac' | 'custom';

// Rate limit strategy
type RateLimitStrategy = 'fixed_window' | 'sliding_window' | 'token_bucket' | 'leaky_bucket';

// API endpoint
interface ApiEndpoint {
  id: string;
  apiId: string;
  path: string;
  method: HttpMethod;
  name: string;
  description: string;
  status: ApiStatus;
  version: string;
  authentication: {
    required: boolean;
    type: AuthType;
    scopes?: string[];
  };
  rateLimit: {
    enabled: boolean;
    requests: number;
    window: number;
    strategy: RateLimitStrategy;
    keyBy?: 'ip' | 'user' | 'api_key' | 'custom';
  };
  request: {
    headers?: { name: string; required: boolean; description: string }[];
    queryParams?: { name: string; type: string; required: boolean; description: string }[];
    pathParams?: { name: string; type: string; description: string }[];
    body?: {
      contentType: string;
      schema?: Record<string, unknown>;
      example?: unknown;
    };
  };
  response: {
    contentType: string;
    statusCodes: {
      code: number;
      description: string;
      schema?: Record<string, unknown>;
      example?: unknown;
    }[];
  };
  backend: {
    type: 'http' | 'lambda' | 'mock' | 'websocket';
    url?: string;
    method?: HttpMethod;
    timeout: number;
    retries: number;
    circuitBreaker?: {
      enabled: boolean;
      threshold: number;
      timeout: number;
    };
  };
  transformation: {
    request?: TransformRule[];
    response?: TransformRule[];
  };
  validation: {
    enabled: boolean;
    validateRequest: boolean;
    validateResponse: boolean;
    strictMode: boolean;
  };
  caching: {
    enabled: boolean;
    ttl: number;
    keyParams?: string[];
    varyHeaders?: string[];
  };
  cors: {
    enabled: boolean;
    allowedOrigins: string[];
    allowedMethods: HttpMethod[];
    allowedHeaders: string[];
    exposeHeaders: string[];
    maxAge: number;
    credentials: boolean;
  };
  logging: {
    enabled: boolean;
    level: 'none' | 'error' | 'info' | 'debug';
    includeRequestBody: boolean;
    includeResponseBody: boolean;
  };
  tags: string[];
  metadata: {
    createdAt: Date;
    createdBy: string;
    updatedAt: Date;
    updatedBy: string;
  };
}

// Transform rule
interface TransformRule {
  id: string;
  type: 'add' | 'remove' | 'rename' | 'modify' | 'map';
  target: 'header' | 'query' | 'body' | 'path';
  source?: string;
  destination?: string;
  value?: unknown;
  template?: string;
  condition?: string;
}

// API definition
interface ApiDefinition {
  id: string;
  name: string;
  description: string;
  version: string;
  basePath: string;
  status: ApiStatus;
  environment: 'development' | 'staging' | 'production';
  endpoints: ApiEndpoint[];
  authentication: {
    defaultType: AuthType;
    apiKeyHeader?: string;
    apiKeyQuery?: string;
    jwtSecret?: string;
    oauth2Config?: {
      authorizationUrl: string;
      tokenUrl: string;
      scopes: { name: string; description: string }[];
    };
  };
  rateLimit: {
    enabled: boolean;
    defaultRequests: number;
    defaultWindow: number;
    strategy: RateLimitStrategy;
  };
  security: {
    ipWhitelist?: string[];
    ipBlacklist?: string[];
    requireHttps: boolean;
    validateOrigin: boolean;
    allowedOrigins?: string[];
  };
  documentation: {
    enabled: boolean;
    url?: string;
    openApiSpec?: string;
  };
  metadata: {
    createdAt: Date;
    createdBy: string;
    updatedAt: Date;
    updatedBy: string;
    publishedAt?: Date;
  };
}

// API key
interface ApiKey {
  id: string;
  name: string;
  key: string;
  prefix: string;
  status: 'active' | 'inactive' | 'revoked' | 'expired';
  owner: {
    type: 'user' | 'application' | 'service';
    id: string;
    name: string;
  };
  permissions: {
    apis: string[];
    endpoints: string[];
    scopes: string[];
  };
  rateLimit: {
    requests: number;
    window: number;
  };
  quota: {
    enabled: boolean;
    limit: number;
    period: 'day' | 'week' | 'month';
    used: number;
    resetAt: Date;
  };
  restrictions: {
    ipAddresses?: string[];
    referrers?: string[];
    environments?: string[];
  };
  metadata: {
    createdAt: Date;
    createdBy: string;
    expiresAt?: Date;
    lastUsedAt?: Date;
    usageCount: number;
  };
}

// Request log
interface RequestLog {
  id: string;
  requestId: string;
  timestamp: Date;
  apiId: string;
  endpointId: string;
  path: string;
  method: HttpMethod;
  clientIp: string;
  userAgent: string;
  apiKeyId?: string;
  userId?: string;
  request: {
    headers: Record<string, string>;
    queryParams: Record<string, string>;
    body?: unknown;
    size: number;
  };
  response: {
    statusCode: number;
    headers: Record<string, string>;
    body?: unknown;
    size: number;
  };
  latency: {
    total: number;
    gateway: number;
    backend: number;
  };
  cache: {
    hit: boolean;
    key?: string;
  };
  rateLimit: {
    remaining: number;
    limit: number;
    reset: Date;
  };
  errors?: {
    type: string;
    message: string;
    stack?: string;
  }[];
  tags: string[];
}

// Rate limit info
interface RateLimitInfo {
  key: string;
  identifier: string;
  requests: number;
  limit: number;
  remaining: number;
  window: number;
  resetAt: Date;
  blocked: boolean;
  blockedUntil?: Date;
}

// Circuit breaker state
interface CircuitBreakerState {
  endpointId: string;
  state: 'closed' | 'open' | 'half_open';
  failures: number;
  successes: number;
  lastFailure?: Date;
  lastSuccess?: Date;
  openedAt?: Date;
  closesAt?: Date;
  threshold: number;
  halfOpenRequests: number;
}

// Gateway metrics
interface GatewayMetrics {
  period: { start: Date; end: Date };
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageLatency: number;
  p50Latency: number;
  p95Latency: number;
  p99Latency: number;
  byStatusCode: Record<string, number>;
  byEndpoint: { endpointId: string; path: string; requests: number; avgLatency: number }[];
  byApiKey: { apiKeyId: string; name: string; requests: number }[];
  cacheHitRate: number;
  rateLimitExceeded: number;
  authFailures: number;
  errors: { type: string; count: number }[];
  bandwidth: {
    inbound: number;
    outbound: number;
  };
}

// Health check
interface HealthCheck {
  endpointId: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  lastCheck: Date;
  nextCheck: Date;
  consecutiveFailures: number;
  consecutiveSuccesses: number;
  latency: number;
  details?: {
    statusCode?: number;
    responseTime?: number;
    error?: string;
  };
}

// Plugin
interface GatewayPlugin {
  id: string;
  name: string;
  type: 'authentication' | 'rate_limiting' | 'transformation' | 'logging' | 'caching' | 'security' | 'custom';
  enabled: boolean;
  priority: number;
  scope: 'global' | 'api' | 'endpoint';
  apiId?: string;
  endpointId?: string;
  config: Record<string, unknown>;
  metadata: {
    createdAt: Date;
    updatedAt: Date;
  };
}

class ApiGatewayService {
  private static instance: ApiGatewayService;
  private apis: Map<string, ApiDefinition> = new Map();
  private apiKeys: Map<string, ApiKey> = new Map();
  private requestLogs: RequestLog[] = [];
  private rateLimits: Map<string, RateLimitInfo> = new Map();
  private circuitBreakers: Map<string, CircuitBreakerState> = new Map();
  private healthChecks: Map<string, HealthCheck> = new Map();
  private plugins: Map<string, GatewayPlugin> = new Map();
  private listeners: ((event: string, data: unknown) => void)[] = [];

  private constructor() {
    this.initializeSampleData();
  }

  public static getInstance(): ApiGatewayService {
    if (!ApiGatewayService.instance) {
      ApiGatewayService.instance = new ApiGatewayService();
    }
    return ApiGatewayService.instance;
  }

  /**
   * Initialize sample data
   */
  private initializeSampleData(): void {
    // Initialize APIs
    const apisData = [
      {
        name: 'Alert API',
        description: 'Public API for alert management and notifications',
        version: 'v1',
        basePath: '/api/v1/alerts',
        status: 'active',
      },
      {
        name: 'Shelter API',
        description: 'API for shelter information and availability',
        version: 'v1',
        basePath: '/api/v1/shelters',
        status: 'active',
      },
      {
        name: 'Resource API',
        description: 'API for resource management and distribution',
        version: 'v1',
        basePath: '/api/v1/resources',
        status: 'active',
      },
      {
        name: 'User API',
        description: 'User management and authentication API',
        version: 'v1',
        basePath: '/api/v1/users',
        status: 'active',
      },
      {
        name: 'Analytics API',
        description: 'Analytics and reporting API (Beta)',
        version: 'v1',
        basePath: '/api/v1/analytics',
        status: 'beta',
      },
    ];

    apisData.forEach((api, idx) => {
      const apiDef: ApiDefinition = {
        id: `api-${(idx + 1).toString().padStart(4, '0')}`,
        name: api.name,
        description: api.description,
        version: api.version,
        basePath: api.basePath,
        status: api.status as ApiStatus,
        environment: 'production',
        endpoints: [],
        authentication: {
          defaultType: 'api_key',
          apiKeyHeader: 'X-API-Key',
        },
        rateLimit: {
          enabled: true,
          defaultRequests: 1000,
          defaultWindow: 3600,
          strategy: 'sliding_window',
        },
        security: {
          requireHttps: true,
          validateOrigin: true,
          allowedOrigins: ['https://alertaid.com', 'https://app.alertaid.com'],
        },
        documentation: {
          enabled: true,
          url: `https://docs.alertaid.com/${api.basePath.split('/')[3]}`,
        },
        metadata: {
          createdAt: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000),
          createdBy: 'admin',
          updatedAt: new Date(),
          updatedBy: 'admin',
          publishedAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
        },
      };

      // Add endpoints
      const endpointTemplates = [
        { path: '', method: 'GET', name: 'List', description: 'List all items' },
        { path: '/{id}', method: 'GET', name: 'Get', description: 'Get item by ID' },
        { path: '', method: 'POST', name: 'Create', description: 'Create new item' },
        { path: '/{id}', method: 'PUT', name: 'Update', description: 'Update item' },
        { path: '/{id}', method: 'DELETE', name: 'Delete', description: 'Delete item' },
      ];

      endpointTemplates.forEach((ep, epIdx) => {
        const endpoint: ApiEndpoint = {
          id: `endpoint-${idx}-${epIdx}`,
          apiId: apiDef.id,
          path: api.basePath + ep.path,
          method: ep.method as HttpMethod,
          name: `${ep.name} ${api.name.replace(' API', '')}`,
          description: `${ep.description} in ${api.name}`,
          status: 'active',
          version: 'v1',
          authentication: {
            required: ep.method !== 'GET',
            type: 'api_key',
            scopes: ep.method === 'DELETE' ? ['admin'] : undefined,
          },
          rateLimit: {
            enabled: true,
            requests: ep.method === 'POST' ? 100 : 1000,
            window: 3600,
            strategy: 'sliding_window',
            keyBy: 'api_key',
          },
          request: {
            headers: [
              { name: 'Content-Type', required: true, description: 'Content type header' },
              { name: 'X-API-Key', required: true, description: 'API key for authentication' },
            ],
            queryParams: ep.method === 'GET' && !ep.path.includes('{id}') ? [
              { name: 'page', type: 'integer', required: false, description: 'Page number' },
              { name: 'limit', type: 'integer', required: false, description: 'Items per page' },
              { name: 'sort', type: 'string', required: false, description: 'Sort field' },
            ] : undefined,
            pathParams: ep.path.includes('{id}') ? [
              { name: 'id', type: 'string', description: 'Resource ID' },
            ] : undefined,
            body: ['POST', 'PUT', 'PATCH'].includes(ep.method) ? {
              contentType: 'application/json',
              schema: { type: 'object' },
              example: { title: 'Example', description: 'Example description' },
            } : undefined,
          },
          response: {
            contentType: 'application/json',
            statusCodes: [
              { code: 200, description: 'Success', example: { success: true } },
              { code: 400, description: 'Bad Request' },
              { code: 401, description: 'Unauthorized' },
              { code: 404, description: 'Not Found' },
              { code: 500, description: 'Internal Server Error' },
            ],
          },
          backend: {
            type: 'http',
            url: `http://backend.internal${api.basePath}${ep.path}`,
            method: ep.method as HttpMethod,
            timeout: 30000,
            retries: 2,
            circuitBreaker: {
              enabled: true,
              threshold: 5,
              timeout: 60000,
            },
          },
          transformation: {
            request: [],
            response: [],
          },
          validation: {
            enabled: true,
            validateRequest: true,
            validateResponse: false,
            strictMode: false,
          },
          caching: {
            enabled: ep.method === 'GET',
            ttl: 300,
            keyParams: ['page', 'limit'],
          },
          cors: {
            enabled: true,
            allowedOrigins: ['*'],
            allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key'],
            exposeHeaders: ['X-RateLimit-Remaining', 'X-RateLimit-Reset'],
            maxAge: 86400,
            credentials: true,
          },
          logging: {
            enabled: true,
            level: 'info',
            includeRequestBody: true,
            includeResponseBody: false,
          },
          tags: [api.name.toLowerCase().replace(' ', '-'), ep.method.toLowerCase()],
          metadata: {
            createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
            createdBy: 'admin',
            updatedAt: new Date(),
            updatedBy: 'admin',
          },
        };
        apiDef.endpoints.push(endpoint);

        // Initialize health check
        this.healthChecks.set(endpoint.id, {
          endpointId: endpoint.id,
          status: 'healthy',
          lastCheck: new Date(),
          nextCheck: new Date(Date.now() + 60000),
          consecutiveFailures: 0,
          consecutiveSuccesses: 10,
          latency: Math.floor(Math.random() * 100) + 50,
        });

        // Initialize circuit breaker
        this.circuitBreakers.set(endpoint.id, {
          endpointId: endpoint.id,
          state: 'closed',
          failures: 0,
          successes: 100,
          lastSuccess: new Date(),
          threshold: 5,
          halfOpenRequests: 0,
        });
      });

      this.apis.set(apiDef.id, apiDef);
    });

    // Initialize API keys
    const apiKeysData = [
      { name: 'Production Mobile App', owner: 'mobile-app' },
      { name: 'Web Dashboard', owner: 'web-app' },
      { name: 'Partner Integration', owner: 'partner-1' },
      { name: 'Internal Service', owner: 'service-auth' },
      { name: 'Testing Key', owner: 'qa-team' },
    ];

    apiKeysData.forEach((ak, idx) => {
      const apiKey: ApiKey = {
        id: `key-${(idx + 1).toString().padStart(4, '0')}`,
        name: ak.name,
        key: `ak_live_${this.generateRandomString(32)}`,
        prefix: 'ak_live_',
        status: idx === 4 ? 'inactive' : 'active',
        owner: {
          type: idx < 2 ? 'application' : idx === 3 ? 'service' : 'user',
          id: ak.owner,
          name: ak.name,
        },
        permissions: {
          apis: ['api-0001', 'api-0002', 'api-0003'],
          endpoints: [],
          scopes: idx === 0 || idx === 1 ? ['read', 'write'] : ['read'],
        },
        rateLimit: {
          requests: idx < 2 ? 10000 : 1000,
          window: 3600,
        },
        quota: {
          enabled: true,
          limit: idx < 2 ? 1000000 : 100000,
          period: 'month',
          used: Math.floor(Math.random() * 50000),
          resetAt: new Date(Date.now() + 15 * 24 * 60 * 60 * 1000),
        },
        restrictions: {},
        metadata: {
          createdAt: new Date(Date.now() - (idx + 1) * 30 * 24 * 60 * 60 * 1000),
          createdBy: 'admin',
          lastUsedAt: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
          usageCount: Math.floor(Math.random() * 100000),
        },
      };
      this.apiKeys.set(apiKey.id, apiKey);
    });

    // Initialize plugins
    const pluginsData = [
      { name: 'Request Logger', type: 'logging', scope: 'global' },
      { name: 'Rate Limiter', type: 'rate_limiting', scope: 'global' },
      { name: 'CORS Handler', type: 'security', scope: 'global' },
      { name: 'Response Cache', type: 'caching', scope: 'api' },
      { name: 'JWT Validator', type: 'authentication', scope: 'endpoint' },
    ];

    pluginsData.forEach((p, idx) => {
      const plugin: GatewayPlugin = {
        id: `plugin-${(idx + 1).toString().padStart(4, '0')}`,
        name: p.name,
        type: p.type as GatewayPlugin['type'],
        enabled: true,
        priority: (idx + 1) * 10,
        scope: p.scope as GatewayPlugin['scope'],
        apiId: p.scope === 'api' ? 'api-0001' : undefined,
        endpointId: p.scope === 'endpoint' ? 'endpoint-0-0' : undefined,
        config: {},
        metadata: {
          createdAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000),
          updatedAt: new Date(),
        },
      };
      this.plugins.set(plugin.id, plugin);
    });

    // Initialize sample request logs
    for (let i = 0; i < 50; i++) {
      const apiIdx = i % 5;
      const endpointIdx = i % 5;
      const log: RequestLog = {
        id: `log-${(i + 1).toString().padStart(6, '0')}`,
        requestId: this.generateRandomString(16),
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
        apiId: `api-${(apiIdx + 1).toString().padStart(4, '0')}`,
        endpointId: `endpoint-${apiIdx}-${endpointIdx}`,
        path: `/api/v1/${['alerts', 'shelters', 'resources', 'users', 'analytics'][apiIdx]}`,
        method: ['GET', 'POST', 'PUT', 'DELETE'][i % 4] as HttpMethod,
        clientIp: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
        userAgent: 'Mozilla/5.0 (compatible; AlertAid/1.0)',
        apiKeyId: `key-${((i % 5) + 1).toString().padStart(4, '0')}`,
        request: {
          headers: { 'Content-Type': 'application/json' },
          queryParams: {},
          size: Math.floor(Math.random() * 1000),
        },
        response: {
          statusCode: [200, 200, 200, 201, 400, 401, 404, 500][i % 8],
          headers: { 'Content-Type': 'application/json' },
          size: Math.floor(Math.random() * 5000),
        },
        latency: {
          total: Math.floor(Math.random() * 500) + 50,
          gateway: Math.floor(Math.random() * 20),
          backend: Math.floor(Math.random() * 400) + 30,
        },
        cache: {
          hit: Math.random() > 0.7,
        },
        rateLimit: {
          remaining: Math.floor(Math.random() * 1000),
          limit: 1000,
          reset: new Date(Date.now() + 3600000),
        },
        tags: [],
      };
      this.requestLogs.push(log);
    }

    // Initialize rate limits
    for (let i = 0; i < 10; i++) {
      const info: RateLimitInfo = {
        key: `rl-${this.generateRandomString(8)}`,
        identifier: `key-${((i % 5) + 1).toString().padStart(4, '0')}`,
        requests: Math.floor(Math.random() * 500),
        limit: 1000,
        remaining: Math.floor(Math.random() * 1000),
        window: 3600,
        resetAt: new Date(Date.now() + Math.random() * 3600000),
        blocked: Math.random() > 0.95,
      };
      this.rateLimits.set(info.key, info);
    }
  }

  /**
   * Generate random string
   */
  private generateRandomString(length: number): string {
    const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    return Array.from({ length }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
  }

  /**
   * Get APIs
   */
  public getApis(filter?: { status?: ApiStatus }): ApiDefinition[] {
    let apis = Array.from(this.apis.values());
    if (filter?.status) apis = apis.filter((a) => a.status === filter.status);
    return apis.sort((a, b) => a.name.localeCompare(b.name));
  }

  /**
   * Get API
   */
  public getApi(id: string): ApiDefinition | undefined {
    return this.apis.get(id);
  }

  /**
   * Get endpoints
   */
  public getEndpoints(apiId?: string): ApiEndpoint[] {
    const apis = Array.from(this.apis.values());
    if (apiId) {
      const api = this.apis.get(apiId);
      return api ? api.endpoints : [];
    }
    return apis.flatMap((a) => a.endpoints);
  }

  /**
   * Get endpoint
   */
  public getEndpoint(id: string): ApiEndpoint | undefined {
    for (const api of this.apis.values()) {
      const endpoint = api.endpoints.find((e) => e.id === id);
      if (endpoint) return endpoint;
    }
    return undefined;
  }

  /**
   * Get API keys
   */
  public getApiKeys(filter?: { status?: ApiKey['status'] }): ApiKey[] {
    let keys = Array.from(this.apiKeys.values());
    if (filter?.status) keys = keys.filter((k) => k.status === filter.status);
    return keys;
  }

  /**
   * Get API key
   */
  public getApiKey(id: string): ApiKey | undefined {
    return this.apiKeys.get(id);
  }

  /**
   * Create API key
   */
  public createApiKey(data: {
    name: string;
    owner: ApiKey['owner'];
    permissions: ApiKey['permissions'];
    rateLimit: ApiKey['rateLimit'];
    expiresIn?: number;
    creator: string;
  }): ApiKey {
    const apiKey: ApiKey = {
      id: `key-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
      name: data.name,
      key: `ak_live_${this.generateRandomString(32)}`,
      prefix: 'ak_live_',
      status: 'active',
      owner: data.owner,
      permissions: data.permissions,
      rateLimit: data.rateLimit,
      quota: {
        enabled: true,
        limit: 100000,
        period: 'month',
        used: 0,
        resetAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
      restrictions: {},
      metadata: {
        createdAt: new Date(),
        createdBy: data.creator,
        expiresAt: data.expiresIn ? new Date(Date.now() + data.expiresIn) : undefined,
        usageCount: 0,
      },
    };

    this.apiKeys.set(apiKey.id, apiKey);
    this.emit('api_key_created', apiKey);

    return apiKey;
  }

  /**
   * Revoke API key
   */
  public revokeApiKey(id: string): void {
    const apiKey = this.apiKeys.get(id);
    if (!apiKey) throw new Error('API key not found');

    apiKey.status = 'revoked';
    this.emit('api_key_revoked', apiKey);
  }

  /**
   * Validate API key
   */
  public validateApiKey(key: string): { valid: boolean; apiKey?: ApiKey; error?: string } {
    for (const apiKey of this.apiKeys.values()) {
      if (apiKey.key === key) {
        if (apiKey.status !== 'active') {
          return { valid: false, error: `API key is ${apiKey.status}` };
        }
        if (apiKey.metadata.expiresAt && apiKey.metadata.expiresAt < new Date()) {
          return { valid: false, error: 'API key has expired' };
        }
        return { valid: true, apiKey };
      }
    }
    return { valid: false, error: 'Invalid API key' };
  }

  /**
   * Check rate limit
   */
  public checkRateLimit(identifier: string, limit: number, window: number): RateLimitInfo {
    const key = `rl-${identifier}`;
    let info = this.rateLimits.get(key);

    if (!info || info.resetAt < new Date()) {
      info = {
        key,
        identifier,
        requests: 0,
        limit,
        remaining: limit,
        window,
        resetAt: new Date(Date.now() + window * 1000),
        blocked: false,
      };
    }

    info.requests++;
    info.remaining = Math.max(0, info.limit - info.requests);
    info.blocked = info.remaining === 0;

    if (info.blocked) {
      info.blockedUntil = info.resetAt;
    }

    this.rateLimits.set(key, info);
    return info;
  }

  /**
   * Get request logs
   */
  public getRequestLogs(filter?: {
    apiId?: string;
    endpointId?: string;
    statusCode?: number[];
    startDate?: Date;
    endDate?: Date;
    limit?: number;
  }): RequestLog[] {
    let logs = [...this.requestLogs];

    if (filter?.apiId) logs = logs.filter((l) => l.apiId === filter.apiId);
    if (filter?.endpointId) logs = logs.filter((l) => l.endpointId === filter.endpointId);
    if (filter?.statusCode?.length) logs = logs.filter((l) => filter.statusCode!.includes(l.response.statusCode));
    if (filter?.startDate) logs = logs.filter((l) => l.timestamp >= filter.startDate!);
    if (filter?.endDate) logs = logs.filter((l) => l.timestamp <= filter.endDate!);

    logs.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    if (filter?.limit) logs = logs.slice(0, filter.limit);

    return logs;
  }

  /**
   * Get metrics
   */
  public getMetrics(period: { start: Date; end: Date }): GatewayMetrics {
    const logs = this.requestLogs.filter(
      (l) => l.timestamp >= period.start && l.timestamp <= period.end
    );

    const totalRequests = logs.length;
    const successfulRequests = logs.filter((l) => l.response.statusCode < 400).length;
    const failedRequests = totalRequests - successfulRequests;

    const latencies = logs.map((l) => l.latency.total).sort((a, b) => a - b);
    const averageLatency = latencies.length > 0
      ? latencies.reduce((a, b) => a + b, 0) / latencies.length
      : 0;

    const byStatusCode: Record<string, number> = {};
    logs.forEach((l) => {
      const code = l.response.statusCode.toString();
      byStatusCode[code] = (byStatusCode[code] || 0) + 1;
    });

    const byEndpoint: { endpointId: string; path: string; requests: number; avgLatency: number }[] = [];
    const endpointGroups = new Map<string, RequestLog[]>();
    logs.forEach((l) => {
      const existing = endpointGroups.get(l.endpointId) || [];
      existing.push(l);
      endpointGroups.set(l.endpointId, existing);
    });

    endpointGroups.forEach((eLogs, endpointId) => {
      const avgLat = eLogs.reduce((a, b) => a + b.latency.total, 0) / eLogs.length;
      byEndpoint.push({
        endpointId,
        path: eLogs[0]?.path || '',
        requests: eLogs.length,
        avgLatency: avgLat,
      });
    });

    return {
      period,
      totalRequests,
      successfulRequests,
      failedRequests,
      averageLatency,
      p50Latency: latencies[Math.floor(latencies.length * 0.5)] || 0,
      p95Latency: latencies[Math.floor(latencies.length * 0.95)] || 0,
      p99Latency: latencies[Math.floor(latencies.length * 0.99)] || 0,
      byStatusCode,
      byEndpoint: byEndpoint.slice(0, 10),
      byApiKey: [],
      cacheHitRate: logs.filter((l) => l.cache.hit).length / (totalRequests || 1),
      rateLimitExceeded: byStatusCode['429'] || 0,
      authFailures: byStatusCode['401'] || 0,
      errors: [],
      bandwidth: {
        inbound: logs.reduce((a, b) => a + b.request.size, 0),
        outbound: logs.reduce((a, b) => a + b.response.size, 0),
      },
    };
  }

  /**
   * Get health checks
   */
  public getHealthChecks(): HealthCheck[] {
    return Array.from(this.healthChecks.values());
  }

  /**
   * Get circuit breaker states
   */
  public getCircuitBreakerStates(): CircuitBreakerState[] {
    return Array.from(this.circuitBreakers.values());
  }

  /**
   * Get plugins
   */
  public getPlugins(): GatewayPlugin[] {
    return Array.from(this.plugins.values())
      .sort((a, b) => a.priority - b.priority);
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

export const apiGatewayService = ApiGatewayService.getInstance();
export type {
  HttpMethod,
  ApiStatus,
  AuthType,
  RateLimitStrategy,
  ApiEndpoint,
  TransformRule,
  ApiDefinition,
  ApiKey,
  RequestLog,
  RateLimitInfo,
  CircuitBreakerState,
  GatewayMetrics,
  HealthCheck,
  GatewayPlugin,
};
