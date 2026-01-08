/**
 * Version Control Service
 * Content versioning, change tracking, rollback capabilities, and diff management
 */

// Version status
type VersionStatus = 'draft' | 'pending_review' | 'approved' | 'published' | 'archived' | 'rejected';

// Change type
type ChangeType = 'create' | 'update' | 'delete' | 'restore' | 'merge' | 'revert';

// Content type
type ContentType = 'alert' | 'shelter' | 'resource' | 'document' | 'config' | 'template' | 'policy' | 'page' | 'component';

// Diff operation
type DiffOperation = 'add' | 'remove' | 'modify' | 'move' | 'unchanged';

// Version
interface Version {
  id: string;
  contentId: string;
  contentType: ContentType;
  versionNumber: string;
  majorVersion: number;
  minorVersion: number;
  patchVersion: number;
  status: VersionStatus;
  title: string;
  description: string;
  content: Record<string, unknown>;
  checksum: string;
  size: number;
  diff?: VersionDiff;
  previousVersionId?: string;
  nextVersionId?: string;
  branch?: string;
  tags: string[];
  metadata: {
    createdAt: Date;
    createdBy: string;
    modifiedAt: Date;
    modifiedBy: string;
    publishedAt?: Date;
    publishedBy?: string;
    archivedAt?: Date;
    archivedBy?: string;
  };
  approval?: {
    status: 'pending' | 'approved' | 'rejected';
    requestedAt: Date;
    requestedBy: string;
    reviewers: string[];
    approvedBy?: string;
    approvedAt?: Date;
    rejectedBy?: string;
    rejectedAt?: Date;
    comments?: string;
  };
  comments: VersionComment[];
}

// Version diff
interface VersionDiff {
  id: string;
  fromVersionId: string;
  toVersionId: string;
  changes: DiffChange[];
  summary: {
    additions: number;
    deletions: number;
    modifications: number;
    unchanged: number;
  };
  createdAt: Date;
}

// Diff change
interface DiffChange {
  path: string;
  operation: DiffOperation;
  oldValue?: unknown;
  newValue?: unknown;
  lineNumber?: number;
  context?: string;
}

// Version comment
interface VersionComment {
  id: string;
  versionId: string;
  author: string;
  authorName: string;
  content: string;
  path?: string;
  lineNumber?: number;
  resolved: boolean;
  resolvedBy?: string;
  resolvedAt?: Date;
  replies: {
    id: string;
    author: string;
    authorName: string;
    content: string;
    createdAt: Date;
  }[];
  createdAt: Date;
  updatedAt: Date;
}

// Branch
interface Branch {
  id: string;
  name: string;
  contentId: string;
  contentType: ContentType;
  description: string;
  baseVersionId: string;
  headVersionId: string;
  status: 'active' | 'merged' | 'abandoned' | 'locked';
  isDefault: boolean;
  isProtected: boolean;
  versions: string[];
  mergeRequests: string[];
  metadata: {
    createdAt: Date;
    createdBy: string;
    updatedAt: Date;
    mergedAt?: Date;
    mergedBy?: string;
  };
}

// Merge request
interface MergeRequest {
  id: string;
  title: string;
  description: string;
  sourceBranchId: string;
  targetBranchId: string;
  status: 'open' | 'merged' | 'closed' | 'conflicted';
  changes: MergeChange[];
  conflicts: MergeConflict[];
  reviewers: {
    userId: string;
    userName: string;
    status: 'pending' | 'approved' | 'changes_requested';
    reviewedAt?: Date;
    comments?: string;
  }[];
  approvalRequired: number;
  approvalCount: number;
  comments: MergeComment[];
  checksPassed: boolean;
  checks: {
    name: string;
    status: 'pending' | 'running' | 'passed' | 'failed';
    description?: string;
    url?: string;
  }[];
  metadata: {
    createdAt: Date;
    createdBy: string;
    updatedAt: Date;
    mergedAt?: Date;
    mergedBy?: string;
    closedAt?: Date;
    closedBy?: string;
  };
}

// Merge change
interface MergeChange {
  path: string;
  operation: DiffOperation;
  additions: number;
  deletions: number;
}

// Merge conflict
interface MergeConflict {
  id: string;
  path: string;
  sourceContent: unknown;
  targetContent: unknown;
  resolvedContent?: unknown;
  resolution?: 'source' | 'target' | 'manual';
  resolvedBy?: string;
  resolvedAt?: Date;
}

// Merge comment
interface MergeComment {
  id: string;
  author: string;
  authorName: string;
  content: string;
  type: 'comment' | 'approval' | 'change_request' | 'system';
  path?: string;
  lineNumber?: number;
  createdAt: Date;
}

// History entry
interface HistoryEntry {
  id: string;
  contentId: string;
  contentType: ContentType;
  versionId: string;
  versionNumber: string;
  changeType: ChangeType;
  description: string;
  actor: string;
  actorName: string;
  timestamp: Date;
  details?: Record<string, unknown>;
  revertible: boolean;
}

// Snapshot
interface Snapshot {
  id: string;
  name: string;
  description: string;
  contentIds: string[];
  versions: { contentId: string; versionId: string }[];
  status: 'active' | 'restoring' | 'restored' | 'expired';
  type: 'manual' | 'automatic' | 'scheduled';
  retentionDays: number;
  size: number;
  metadata: {
    createdAt: Date;
    createdBy: string;
    expiresAt?: Date;
    restoredAt?: Date;
    restoredBy?: string;
  };
}

// Rollback request
interface RollbackRequest {
  id: string;
  contentId: string;
  contentType: ContentType;
  fromVersionId: string;
  toVersionId: string;
  reason: string;
  status: 'pending' | 'approved' | 'executing' | 'completed' | 'failed' | 'cancelled';
  approvalRequired: boolean;
  approver?: string;
  approvedAt?: Date;
  executedAt?: Date;
  executedBy?: string;
  result?: {
    success: boolean;
    newVersionId?: string;
    error?: string;
  };
  metadata: {
    createdAt: Date;
    createdBy: string;
    updatedAt: Date;
  };
}

// Version settings
interface VersionSettings {
  contentType: ContentType;
  maxVersions: number;
  retentionDays: number;
  autoArchive: boolean;
  archiveAfterDays: number;
  requireApproval: boolean;
  minApprovers: number;
  allowedApprovers: string[];
  protectedBranches: string[];
  defaultBranch: string;
  enableComments: boolean;
  enableTags: boolean;
  enableSnapshots: boolean;
  snapshotFrequency?: 'hourly' | 'daily' | 'weekly' | 'monthly';
}

class VersionControlService {
  private static instance: VersionControlService;
  private versions: Map<string, Version> = new Map();
  private branches: Map<string, Branch> = new Map();
  private mergeRequests: Map<string, MergeRequest> = new Map();
  private history: Map<string, HistoryEntry[]> = new Map();
  private snapshots: Map<string, Snapshot> = new Map();
  private rollbacks: Map<string, RollbackRequest> = new Map();
  private settings: Map<ContentType, VersionSettings> = new Map();
  private listeners: ((event: string, data: unknown) => void)[] = [];

  private constructor() {
    this.initializeSampleData();
  }

  public static getInstance(): VersionControlService {
    if (!VersionControlService.instance) {
      VersionControlService.instance = new VersionControlService();
    }
    return VersionControlService.instance;
  }

  /**
   * Initialize sample data
   */
  private initializeSampleData(): void {
    // Initialize settings for each content type
    const contentTypes: ContentType[] = ['alert', 'shelter', 'resource', 'document', 'config', 'template', 'policy', 'page', 'component'];

    contentTypes.forEach((type) => {
      this.settings.set(type, {
        contentType: type,
        maxVersions: 100,
        retentionDays: 365,
        autoArchive: true,
        archiveAfterDays: 90,
        requireApproval: ['policy', 'config', 'template'].includes(type),
        minApprovers: 1,
        allowedApprovers: ['admin', 'manager'],
        protectedBranches: ['main', 'production'],
        defaultBranch: 'main',
        enableComments: true,
        enableTags: true,
        enableSnapshots: true,
        snapshotFrequency: 'daily',
      });
    });

    // Initialize sample versions for alerts
    const alertVersions = [
      { title: 'Cyclone Warning', majorVersion: 2, minorVersion: 1, status: 'published' },
      { title: 'Cyclone Warning', majorVersion: 2, minorVersion: 0, status: 'archived' },
      { title: 'Cyclone Warning', majorVersion: 1, minorVersion: 0, status: 'archived' },
      { title: 'Flood Alert Template', majorVersion: 1, minorVersion: 2, status: 'published' },
      { title: 'Earthquake Response', majorVersion: 3, minorVersion: 0, status: 'draft' },
    ];

    alertVersions.forEach((av, idx) => {
      const version: Version = {
        id: `version-alert-${(idx + 1).toString().padStart(4, '0')}`,
        contentId: `alert-${Math.floor(idx / 3) + 1}`,
        contentType: 'alert',
        versionNumber: `${av.majorVersion}.${av.minorVersion}.0`,
        majorVersion: av.majorVersion,
        minorVersion: av.minorVersion,
        patchVersion: 0,
        status: av.status as VersionStatus,
        title: av.title,
        description: `Version ${av.majorVersion}.${av.minorVersion} of ${av.title}`,
        content: {
          title: av.title,
          severity: ['critical', 'high', 'medium'][idx % 3],
          message: `Alert message content for ${av.title}`,
          regions: ['Tamil Nadu', 'Kerala', 'Karnataka'],
          instructions: ['Stay indoors', 'Follow emergency protocols'],
        },
        checksum: this.generateChecksum(`content-${idx}`),
        size: Math.floor(Math.random() * 10000) + 1000,
        previousVersionId: idx > 0 && idx % 3 !== 0 ? `version-alert-${idx.toString().padStart(4, '0')}` : undefined,
        branch: 'main',
        tags: av.status === 'published' ? ['released', 'stable'] : [],
        metadata: {
          createdAt: new Date(Date.now() - (idx + 1) * 7 * 24 * 60 * 60 * 1000),
          createdBy: 'content-team',
          modifiedAt: new Date(Date.now() - idx * 24 * 60 * 60 * 1000),
          modifiedBy: 'editor-1',
          publishedAt: av.status === 'published' ? new Date(Date.now() - idx * 24 * 60 * 60 * 1000) : undefined,
          publishedBy: av.status === 'published' ? 'publisher-1' : undefined,
        },
        comments: [],
      };

      // Add diff for non-first versions
      if (version.previousVersionId) {
        version.diff = {
          id: `diff-${idx}`,
          fromVersionId: version.previousVersionId,
          toVersionId: version.id,
          changes: [
            { path: 'message', operation: 'modify', oldValue: 'Previous message', newValue: 'Updated message' },
            { path: 'severity', operation: 'modify', oldValue: 'high', newValue: 'critical' },
          ],
          summary: { additions: 5, deletions: 2, modifications: 3, unchanged: 10 },
          createdAt: new Date(),
        };
      }

      this.versions.set(version.id, version);
    });

    // Initialize sample versions for documents
    const docVersions = [
      { title: 'Emergency Procedures', majorVersion: 4, minorVersion: 0, status: 'published' },
      { title: 'Volunteer Guidelines', majorVersion: 2, minorVersion: 3, status: 'approved' },
      { title: 'Safety Manual', majorVersion: 1, minorVersion: 1, status: 'pending_review' },
    ];

    docVersions.forEach((dv, idx) => {
      const version: Version = {
        id: `version-doc-${(idx + 1).toString().padStart(4, '0')}`,
        contentId: `document-${idx + 1}`,
        contentType: 'document',
        versionNumber: `${dv.majorVersion}.${dv.minorVersion}.0`,
        majorVersion: dv.majorVersion,
        minorVersion: dv.minorVersion,
        patchVersion: 0,
        status: dv.status as VersionStatus,
        title: dv.title,
        description: `Version ${dv.majorVersion}.${dv.minorVersion} of ${dv.title}`,
        content: {
          title: dv.title,
          sections: ['Introduction', 'Procedures', 'Appendix'],
          format: 'markdown',
          wordCount: Math.floor(Math.random() * 5000) + 500,
        },
        checksum: this.generateChecksum(`doc-${idx}`),
        size: Math.floor(Math.random() * 50000) + 5000,
        branch: 'main',
        tags: ['documentation', dv.majorVersion > 2 ? 'stable' : 'beta'],
        metadata: {
          createdAt: new Date(Date.now() - (idx + 1) * 14 * 24 * 60 * 60 * 1000),
          createdBy: 'doc-team',
          modifiedAt: new Date(Date.now() - idx * 3 * 24 * 60 * 60 * 1000),
          modifiedBy: 'writer-1',
        },
        approval: dv.status === 'pending_review' ? {
          status: 'pending',
          requestedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
          requestedBy: 'writer-1',
          reviewers: ['reviewer-1', 'reviewer-2'],
        } : undefined,
        comments: [],
      };
      this.versions.set(version.id, version);
    });

    // Initialize branches
    const branchesData = [
      { name: 'main', contentId: 'alert-1', isDefault: true, isProtected: true },
      { name: 'feature/new-alerts', contentId: 'alert-1', isDefault: false, isProtected: false },
      { name: 'main', contentId: 'document-1', isDefault: true, isProtected: true },
      { name: 'draft/updated-procedures', contentId: 'document-1', isDefault: false, isProtected: false },
    ];

    branchesData.forEach((b, idx) => {
      const branch: Branch = {
        id: `branch-${(idx + 1).toString().padStart(4, '0')}`,
        name: b.name,
        contentId: b.contentId,
        contentType: b.contentId.startsWith('alert') ? 'alert' : 'document',
        description: `${b.name} branch for ${b.contentId}`,
        baseVersionId: `version-${b.contentId.startsWith('alert') ? 'alert' : 'doc'}-0001`,
        headVersionId: `version-${b.contentId.startsWith('alert') ? 'alert' : 'doc'}-000${idx + 1}`,
        status: b.name === 'main' ? 'active' : 'active',
        isDefault: b.isDefault,
        isProtected: b.isProtected,
        versions: [],
        mergeRequests: [],
        metadata: {
          createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          createdBy: 'admin',
          updatedAt: new Date(),
        },
      };
      this.branches.set(branch.id, branch);
    });

    // Initialize merge requests
    const mergeRequestsData = [
      { title: 'Add new cyclone warning template', status: 'open' },
      { title: 'Update emergency procedures v4.1', status: 'merged' },
      { title: 'Fix typos in volunteer guidelines', status: 'closed' },
    ];

    mergeRequestsData.forEach((mr, idx) => {
      const mergeRequest: MergeRequest = {
        id: `mr-${(idx + 1).toString().padStart(4, '0')}`,
        title: mr.title,
        description: `Merge request for ${mr.title}`,
        sourceBranchId: `branch-${(idx * 2 + 2).toString().padStart(4, '0')}`,
        targetBranchId: `branch-${(idx * 2 + 1).toString().padStart(4, '0')}`,
        status: mr.status as MergeRequest['status'],
        changes: [
          { path: 'content.message', operation: 'modify', additions: 10, deletions: 5 },
          { path: 'content.regions', operation: 'add', additions: 3, deletions: 0 },
        ],
        conflicts: [],
        reviewers: [
          { userId: 'reviewer-1', userName: 'Reviewer One', status: mr.status === 'merged' ? 'approved' : 'pending' },
          { userId: 'reviewer-2', userName: 'Reviewer Two', status: 'pending' },
        ],
        approvalRequired: 1,
        approvalCount: mr.status === 'merged' ? 1 : 0,
        comments: [
          {
            id: `comment-${idx}-1`,
            author: 'reviewer-1',
            authorName: 'Reviewer One',
            content: 'Looks good overall',
            type: 'comment',
            createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
          },
        ],
        checksPassed: mr.status === 'merged',
        checks: [
          { name: 'Validation', status: 'passed', description: 'Content validation passed' },
          { name: 'Spell Check', status: 'passed', description: 'No spelling errors found' },
        ],
        metadata: {
          createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          createdBy: 'editor-1',
          updatedAt: new Date(),
          mergedAt: mr.status === 'merged' ? new Date(Date.now() - 1 * 24 * 60 * 60 * 1000) : undefined,
          mergedBy: mr.status === 'merged' ? 'reviewer-1' : undefined,
        },
      };
      this.mergeRequests.set(mergeRequest.id, mergeRequest);
    });

    // Initialize snapshots
    const snapshotsData = [
      { name: 'Daily Backup - Today', type: 'automatic' },
      { name: 'Pre-deployment Snapshot', type: 'manual' },
      { name: 'Weekly Backup', type: 'scheduled' },
    ];

    snapshotsData.forEach((s, idx) => {
      const snapshot: Snapshot = {
        id: `snapshot-${(idx + 1).toString().padStart(4, '0')}`,
        name: s.name,
        description: `${s.type} snapshot`,
        contentIds: ['alert-1', 'alert-2', 'document-1'],
        versions: [
          { contentId: 'alert-1', versionId: 'version-alert-0001' },
          { contentId: 'alert-2', versionId: 'version-alert-0004' },
          { contentId: 'document-1', versionId: 'version-doc-0001' },
        ],
        status: 'active',
        type: s.type as Snapshot['type'],
        retentionDays: s.type === 'manual' ? 365 : 30,
        size: Math.floor(Math.random() * 1000000) + 100000,
        metadata: {
          createdAt: new Date(Date.now() - idx * 24 * 60 * 60 * 1000),
          createdBy: s.type === 'manual' ? 'admin' : 'system',
          expiresAt: new Date(Date.now() + (s.type === 'manual' ? 365 : 30) * 24 * 60 * 60 * 1000),
        },
      };
      this.snapshots.set(snapshot.id, snapshot);
    });

    // Initialize history
    const historyEntries: HistoryEntry[] = [
      {
        id: 'history-001',
        contentId: 'alert-1',
        contentType: 'alert',
        versionId: 'version-alert-0001',
        versionNumber: '2.1.0',
        changeType: 'update',
        description: 'Updated cyclone warning content',
        actor: 'editor-1',
        actorName: 'Editor One',
        timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
        revertible: true,
      },
      {
        id: 'history-002',
        contentId: 'alert-1',
        contentType: 'alert',
        versionId: 'version-alert-0002',
        versionNumber: '2.0.0',
        changeType: 'create',
        description: 'Created new major version',
        actor: 'editor-1',
        actorName: 'Editor One',
        timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        revertible: true,
      },
      {
        id: 'history-003',
        contentId: 'document-1',
        contentType: 'document',
        versionId: 'version-doc-0001',
        versionNumber: '4.0.0',
        changeType: 'update',
        description: 'Published emergency procedures v4',
        actor: 'doc-team',
        actorName: 'Documentation Team',
        timestamp: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
        revertible: true,
      },
    ];

    historyEntries.forEach((entry) => {
      const existing = this.history.get(entry.contentId) || [];
      existing.push(entry);
      this.history.set(entry.contentId, existing);
    });
  }

  /**
   * Generate checksum
   */
  private generateChecksum(content: string): string {
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(8, '0');
  }

  /**
   * Get versions
   */
  public getVersions(filter?: {
    contentId?: string;
    contentType?: ContentType;
    status?: VersionStatus[];
    branch?: string;
  }): Version[] {
    let versions = Array.from(this.versions.values());

    if (filter?.contentId) versions = versions.filter((v) => v.contentId === filter.contentId);
    if (filter?.contentType) versions = versions.filter((v) => v.contentType === filter.contentType);
    if (filter?.status?.length) versions = versions.filter((v) => filter.status!.includes(v.status));
    if (filter?.branch) versions = versions.filter((v) => v.branch === filter.branch);

    return versions.sort((a, b) => {
      if (a.majorVersion !== b.majorVersion) return b.majorVersion - a.majorVersion;
      if (a.minorVersion !== b.minorVersion) return b.minorVersion - a.minorVersion;
      return b.patchVersion - a.patchVersion;
    });
  }

  /**
   * Get version
   */
  public getVersion(id: string): Version | undefined {
    return this.versions.get(id);
  }

  /**
   * Create version
   */
  public createVersion(data: {
    contentId: string;
    contentType: ContentType;
    content: Record<string, unknown>;
    title: string;
    description: string;
    versionType: 'major' | 'minor' | 'patch';
    branch?: string;
    author: string;
  }): Version {
    // Get latest version for this content
    const existingVersions = this.getVersions({ contentId: data.contentId });
    const latestVersion = existingVersions[0];

    let majorVersion = 1;
    let minorVersion = 0;
    let patchVersion = 0;

    if (latestVersion) {
      switch (data.versionType) {
        case 'major':
          majorVersion = latestVersion.majorVersion + 1;
          minorVersion = 0;
          patchVersion = 0;
          break;
        case 'minor':
          majorVersion = latestVersion.majorVersion;
          minorVersion = latestVersion.minorVersion + 1;
          patchVersion = 0;
          break;
        case 'patch':
          majorVersion = latestVersion.majorVersion;
          minorVersion = latestVersion.minorVersion;
          patchVersion = latestVersion.patchVersion + 1;
          break;
      }
    }

    const versionNumber = `${majorVersion}.${minorVersion}.${patchVersion}`;
    const contentStr = JSON.stringify(data.content);

    const version: Version = {
      id: `version-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
      contentId: data.contentId,
      contentType: data.contentType,
      versionNumber,
      majorVersion,
      minorVersion,
      patchVersion,
      status: 'draft',
      title: data.title,
      description: data.description,
      content: data.content,
      checksum: this.generateChecksum(contentStr),
      size: contentStr.length,
      previousVersionId: latestVersion?.id,
      branch: data.branch || 'main',
      tags: [],
      metadata: {
        createdAt: new Date(),
        createdBy: data.author,
        modifiedAt: new Date(),
        modifiedBy: data.author,
      },
      comments: [],
    };

    // Generate diff if there's a previous version
    if (latestVersion) {
      version.diff = this.generateDiff(latestVersion.id, version.id, latestVersion.content, data.content);
    }

    this.versions.set(version.id, version);
    this.addHistoryEntry(data.contentId, data.contentType, version.id, versionNumber, 'create', `Created version ${versionNumber}`, data.author);
    this.emit('version_created', version);

    return version;
  }

  /**
   * Generate diff
   */
  private generateDiff(
    fromVersionId: string,
    toVersionId: string,
    oldContent: Record<string, unknown>,
    newContent: Record<string, unknown>
  ): VersionDiff {
    const changes: DiffChange[] = [];
    const allKeys = new Set([...Object.keys(oldContent), ...Object.keys(newContent)]);

    let additions = 0;
    let deletions = 0;
    let modifications = 0;
    let unchanged = 0;

    allKeys.forEach((key) => {
      const oldValue = oldContent[key];
      const newValue = newContent[key];

      if (oldValue === undefined) {
        changes.push({ path: key, operation: 'add', newValue });
        additions++;
      } else if (newValue === undefined) {
        changes.push({ path: key, operation: 'remove', oldValue });
        deletions++;
      } else if (JSON.stringify(oldValue) !== JSON.stringify(newValue)) {
        changes.push({ path: key, operation: 'modify', oldValue, newValue });
        modifications++;
      } else {
        unchanged++;
      }
    });

    return {
      id: `diff-${Date.now()}`,
      fromVersionId,
      toVersionId,
      changes,
      summary: { additions, deletions, modifications, unchanged },
      createdAt: new Date(),
    };
  }

  /**
   * Add history entry
   */
  private addHistoryEntry(
    contentId: string,
    contentType: ContentType,
    versionId: string,
    versionNumber: string,
    changeType: ChangeType,
    description: string,
    actor: string
  ): void {
    const entry: HistoryEntry = {
      id: `history-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
      contentId,
      contentType,
      versionId,
      versionNumber,
      changeType,
      description,
      actor,
      actorName: actor,
      timestamp: new Date(),
      revertible: true,
    };

    const existing = this.history.get(contentId) || [];
    existing.unshift(entry);
    this.history.set(contentId, existing);
  }

  /**
   * Update version status
   */
  public updateVersionStatus(id: string, status: VersionStatus, actor: string): Version {
    const version = this.versions.get(id);
    if (!version) throw new Error('Version not found');

    const oldStatus = version.status;
    version.status = status;
    version.metadata.modifiedAt = new Date();
    version.metadata.modifiedBy = actor;

    if (status === 'published') {
      version.metadata.publishedAt = new Date();
      version.metadata.publishedBy = actor;
    } else if (status === 'archived') {
      version.metadata.archivedAt = new Date();
      version.metadata.archivedBy = actor;
    }

    this.addHistoryEntry(version.contentId, version.contentType, version.id, version.versionNumber, 'update', `Status changed from ${oldStatus} to ${status}`, actor);
    this.emit('version_status_changed', { version, oldStatus, newStatus: status });

    return version;
  }

  /**
   * Get branches
   */
  public getBranches(contentId?: string): Branch[] {
    let branches = Array.from(this.branches.values());
    if (contentId) branches = branches.filter((b) => b.contentId === contentId);
    return branches;
  }

  /**
   * Get branch
   */
  public getBranch(id: string): Branch | undefined {
    return this.branches.get(id);
  }

  /**
   * Create branch
   */
  public createBranch(data: {
    name: string;
    contentId: string;
    contentType: ContentType;
    description: string;
    baseVersionId: string;
    author: string;
  }): Branch {
    const branch: Branch = {
      id: `branch-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
      name: data.name,
      contentId: data.contentId,
      contentType: data.contentType,
      description: data.description,
      baseVersionId: data.baseVersionId,
      headVersionId: data.baseVersionId,
      status: 'active',
      isDefault: false,
      isProtected: false,
      versions: [data.baseVersionId],
      mergeRequests: [],
      metadata: {
        createdAt: new Date(),
        createdBy: data.author,
        updatedAt: new Date(),
      },
    };

    this.branches.set(branch.id, branch);
    this.emit('branch_created', branch);

    return branch;
  }

  /**
   * Get merge requests
   */
  public getMergeRequests(filter?: { status?: MergeRequest['status'][] }): MergeRequest[] {
    let requests = Array.from(this.mergeRequests.values());
    if (filter?.status?.length) requests = requests.filter((r) => filter.status!.includes(r.status));
    return requests.sort((a, b) => b.metadata.updatedAt.getTime() - a.metadata.updatedAt.getTime());
  }

  /**
   * Get merge request
   */
  public getMergeRequest(id: string): MergeRequest | undefined {
    return this.mergeRequests.get(id);
  }

  /**
   * Create merge request
   */
  public createMergeRequest(data: {
    title: string;
    description: string;
    sourceBranchId: string;
    targetBranchId: string;
    reviewers: string[];
    author: string;
  }): MergeRequest {
    const mergeRequest: MergeRequest = {
      id: `mr-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
      title: data.title,
      description: data.description,
      sourceBranchId: data.sourceBranchId,
      targetBranchId: data.targetBranchId,
      status: 'open',
      changes: [],
      conflicts: [],
      reviewers: data.reviewers.map((r) => ({
        userId: r,
        userName: r,
        status: 'pending' as const,
      })),
      approvalRequired: 1,
      approvalCount: 0,
      comments: [],
      checksPassed: false,
      checks: [],
      metadata: {
        createdAt: new Date(),
        createdBy: data.author,
        updatedAt: new Date(),
      },
    };

    this.mergeRequests.set(mergeRequest.id, mergeRequest);
    this.emit('merge_request_created', mergeRequest);

    return mergeRequest;
  }

  /**
   * Get history
   */
  public getHistory(contentId: string): HistoryEntry[] {
    return this.history.get(contentId) || [];
  }

  /**
   * Get snapshots
   */
  public getSnapshots(): Snapshot[] {
    return Array.from(this.snapshots.values())
      .sort((a, b) => b.metadata.createdAt.getTime() - a.metadata.createdAt.getTime());
  }

  /**
   * Create snapshot
   */
  public createSnapshot(data: {
    name: string;
    description: string;
    contentIds: string[];
    retentionDays: number;
    author: string;
  }): Snapshot {
    const versions: Snapshot['versions'] = [];

    data.contentIds.forEach((contentId) => {
      const contentVersions = this.getVersions({ contentId });
      if (contentVersions.length > 0) {
        versions.push({ contentId, versionId: contentVersions[0].id });
      }
    });

    const snapshot: Snapshot = {
      id: `snapshot-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
      name: data.name,
      description: data.description,
      contentIds: data.contentIds,
      versions,
      status: 'active',
      type: 'manual',
      retentionDays: data.retentionDays,
      size: versions.length * 10000,
      metadata: {
        createdAt: new Date(),
        createdBy: data.author,
        expiresAt: new Date(Date.now() + data.retentionDays * 24 * 60 * 60 * 1000),
      },
    };

    this.snapshots.set(snapshot.id, snapshot);
    this.emit('snapshot_created', snapshot);

    return snapshot;
  }

  /**
   * Rollback to version
   */
  public rollback(data: {
    contentId: string;
    toVersionId: string;
    reason: string;
    author: string;
  }): RollbackRequest {
    const targetVersion = this.versions.get(data.toVersionId);
    if (!targetVersion) throw new Error('Target version not found');

    const currentVersions = this.getVersions({ contentId: data.contentId });
    const currentVersion = currentVersions[0];
    if (!currentVersion) throw new Error('No current version found');

    const rollbackRequest: RollbackRequest = {
      id: `rollback-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
      contentId: data.contentId,
      contentType: targetVersion.contentType,
      fromVersionId: currentVersion.id,
      toVersionId: data.toVersionId,
      reason: data.reason,
      status: 'pending',
      approvalRequired: true,
      metadata: {
        createdAt: new Date(),
        createdBy: data.author,
        updatedAt: new Date(),
      },
    };

    this.rollbacks.set(rollbackRequest.id, rollbackRequest);
    this.emit('rollback_requested', rollbackRequest);

    return rollbackRequest;
  }

  /**
   * Get settings
   */
  public getSettings(contentType: ContentType): VersionSettings | undefined {
    return this.settings.get(contentType);
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

export const versionControlService = VersionControlService.getInstance();
export type {
  VersionStatus,
  ChangeType,
  ContentType,
  DiffOperation,
  Version,
  VersionDiff,
  DiffChange,
  VersionComment,
  Branch,
  MergeRequest,
  MergeChange,
  MergeConflict,
  MergeComment,
  HistoryEntry,
  Snapshot,
  RollbackRequest,
  VersionSettings,
};
