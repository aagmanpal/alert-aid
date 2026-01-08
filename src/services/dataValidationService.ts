/**
 * Crowd-sourced Data Validation Service with ML
 * Validates user-submitted disaster reports using machine learning
 * Detects fake/misleading reports and assigns credibility scores
 */

// Report data interface
interface ReportData {
  id: string;
  userId: string;
  timestamp: Date;
  location: {
    latitude: number;
    longitude: number;
    address?: string;
  };
  disasterType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  images?: string[];
  videos?: string[];
  witnesses?: number;
}

// User reputation tracking
interface UserReputation {
  userId: string;
  totalReports: number;
  verifiedReports: number;
  flaggedReports: number;
  averageAccuracy: number;
  trustScore: number;
  registrationDate: Date;
  lastActiveDate: Date;
  badges: string[];
}

// Validation result
interface ValidationResult {
  reportId: string;
  isValid: boolean;
  credibilityScore: number;
  confidence: number;
  flags: ValidationFlag[];
  crossReferences: CrossReference[];
  recommendations: string[];
  verificationStatus: 'pending' | 'verified' | 'disputed' | 'rejected';
}

// Validation flags
interface ValidationFlag {
  type: 'location_mismatch' | 'time_anomaly' | 'duplicate' | 'spam' | 'fake_image' | 'inconsistent' | 'user_reputation';
  severity: 'low' | 'medium' | 'high';
  description: string;
  evidence: string;
}

// Cross-reference with other reports
interface CrossReference {
  reportId: string;
  similarity: number;
  distance: number;
  timeDifference: number;
  userId: string;
  supports: boolean;
}

// ML Model weights for validation
const VALIDATION_MODEL = {
  userReputationWeight: 0.25,
  locationConsistencyWeight: 0.20,
  temporalConsistencyWeight: 0.15,
  contentAnalysisWeight: 0.20,
  crossReferenceWeight: 0.20,
  
  // Spam detection patterns
  spamPatterns: [
    /\b(click here|free money|winner|lottery)\b/gi,
    /\b(urgent|act now|limited time)\b/gi,
    /(.)\1{5,}/g, // Repeated characters
    /\b[A-Z]{10,}\b/g, // All caps words
  ],
  
  // Credibility thresholds
  thresholds: {
    highCredibility: 0.8,
    mediumCredibility: 0.5,
    lowCredibility: 0.3,
    reject: 0.2,
  },
};

// Known disaster patterns for ML validation
const DISASTER_PATTERNS: Record<string, {
  keywords: string[];
  typicalDuration: number; // hours
  spreadRadius: number; // km
  minWitnesses: number;
}> = {
  flood: {
    keywords: ['water', 'submerged', 'rising', 'flooded', 'waterlogged', 'overflow', 'dam', 'rain'],
    typicalDuration: 72,
    spreadRadius: 50,
    minWitnesses: 3,
  },
  earthquake: {
    keywords: ['shaking', 'tremor', 'collapsed', 'magnitude', 'epicenter', 'aftershock', 'seismic'],
    typicalDuration: 1,
    spreadRadius: 200,
    minWitnesses: 10,
  },
  cyclone: {
    keywords: ['wind', 'storm', 'landfall', 'eye', 'evacuation', 'coastal', 'surge'],
    typicalDuration: 48,
    spreadRadius: 300,
    minWitnesses: 20,
  },
  wildfire: {
    keywords: ['fire', 'smoke', 'burning', 'spread', 'evacuate', 'ash', 'flames', 'forest'],
    typicalDuration: 120,
    spreadRadius: 30,
    minWitnesses: 5,
  },
  landslide: {
    keywords: ['slide', 'mud', 'debris', 'slope', 'collapsed', 'buried', 'hillside'],
    typicalDuration: 2,
    spreadRadius: 5,
    minWitnesses: 2,
  },
};

// Simulated user reputation database
const USER_REPUTATIONS: Map<string, UserReputation> = new Map();

// Report storage for cross-referencing
const REPORT_STORAGE: Map<string, ReportData> = new Map();

class DataValidationService {
  private static instance: DataValidationService;

  private constructor() {
    this.initializeSampleData();
  }

  public static getInstance(): DataValidationService {
    if (!DataValidationService.instance) {
      DataValidationService.instance = new DataValidationService();
    }
    return DataValidationService.instance;
  }

  /**
   * Main validation method - validates a crowd-sourced report
   */
  public async validateReport(report: ReportData): Promise<ValidationResult> {
    const flags: ValidationFlag[] = [];
    const crossReferences: CrossReference[] = [];
    
    // Get or create user reputation
    const userReputation = this.getUserReputation(report.userId);
    
    // 1. User Reputation Analysis
    const reputationScore = this.analyzeUserReputation(userReputation, flags);
    
    // 2. Location Consistency Check
    const locationScore = await this.validateLocation(report, flags);
    
    // 3. Temporal Consistency Check
    const temporalScore = this.validateTemporal(report, flags);
    
    // 4. Content Analysis (NLP-based)
    const contentScore = this.analyzeContent(report, flags);
    
    // 5. Cross-Reference with Other Reports
    const crossRefScore = this.crossReferenceReports(report, crossReferences, flags);
    
    // 6. Image Analysis (if present)
    let imageScore = 1.0;
    if (report.images && report.images.length > 0) {
      imageScore = await this.analyzeImages(report.images, flags);
    }
    
    // Calculate weighted credibility score
    const credibilityScore = this.calculateCredibilityScore({
      reputationScore,
      locationScore,
      temporalScore,
      contentScore,
      crossRefScore,
      imageScore,
    });
    
    // Determine verification status
    const verificationStatus = this.determineVerificationStatus(credibilityScore, flags);
    
    // Generate recommendations
    const recommendations = this.generateRecommendations(credibilityScore, flags, userReputation);
    
    // Store report for future cross-referencing
    REPORT_STORAGE.set(report.id, report);
    
    // Update user reputation based on validation
    this.updateUserReputation(report.userId, credibilityScore);
    
    return {
      reportId: report.id,
      isValid: credibilityScore >= VALIDATION_MODEL.thresholds.lowCredibility,
      credibilityScore,
      confidence: this.calculateConfidence(flags, crossReferences),
      flags,
      crossReferences,
      recommendations,
      verificationStatus,
    };
  }

  /**
   * Analyze user reputation
   */
  private analyzeUserReputation(reputation: UserReputation, flags: ValidationFlag[]): number {
    let score = reputation.trustScore;
    
    // New user penalty
    const accountAge = Date.now() - reputation.registrationDate.getTime();
    const daysOld = accountAge / (1000 * 60 * 60 * 24);
    
    if (daysOld < 7) {
      score *= 0.5;
      flags.push({
        type: 'user_reputation',
        severity: 'medium',
        description: 'New user account',
        evidence: `Account is only ${Math.round(daysOld)} days old`,
      });
    }
    
    // Check flagged reports ratio
    if (reputation.totalReports > 5) {
      const flaggedRatio = reputation.flaggedReports / reputation.totalReports;
      if (flaggedRatio > 0.3) {
        score *= 0.6;
        flags.push({
          type: 'user_reputation',
          severity: 'high',
          description: 'High ratio of flagged reports',
          evidence: `${Math.round(flaggedRatio * 100)}% of reports have been flagged`,
        });
      }
    }
    
    // Bonus for verified reports
    if (reputation.verifiedReports > 10 && reputation.averageAccuracy > 0.8) {
      score = Math.min(1.0, score * 1.2);
    }
    
    return Math.max(0, Math.min(1, score));
  }

  /**
   * Validate location consistency
   */
  private async validateLocation(report: ReportData, flags: ValidationFlag[]): Promise<number> {
    let score = 1.0;
    
    // Check if coordinates are valid
    if (report.location.latitude < -90 || report.location.latitude > 90 ||
        report.location.longitude < -180 || report.location.longitude > 180) {
      score = 0;
      flags.push({
        type: 'location_mismatch',
        severity: 'high',
        description: 'Invalid coordinates',
        evidence: `Lat: ${report.location.latitude}, Lng: ${report.location.longitude}`,
      });
      return score;
    }
    
    // Check if location matches address (if provided)
    if (report.location.address) {
      const addressMatch = this.validateAddressConsistency(
        report.location.latitude,
        report.location.longitude,
        report.location.address
      );
      if (!addressMatch) {
        score *= 0.7;
        flags.push({
          type: 'location_mismatch',
          severity: 'medium',
          description: 'Address does not match coordinates',
          evidence: `Address: ${report.location.address}`,
        });
      }
    }
    
    // Check if location is in a known populated area
    const isPopulatedArea = this.checkPopulatedArea(report.location.latitude, report.location.longitude);
    if (!isPopulatedArea) {
      score *= 0.8;
      flags.push({
        type: 'location_mismatch',
        severity: 'low',
        description: 'Location is in a sparsely populated area',
        evidence: 'Few residents to verify report',
      });
    }
    
    return score;
  }

  /**
   * Validate temporal consistency
   */
  private validateTemporal(report: ReportData, flags: ValidationFlag[]): number {
    let score = 1.0;
    const now = Date.now();
    const reportTime = report.timestamp.getTime();
    
    // Check if report is from the future
    if (reportTime > now + 60000) { // Allow 1 minute tolerance
      score = 0;
      flags.push({
        type: 'time_anomaly',
        severity: 'high',
        description: 'Report timestamp is in the future',
        evidence: `Report time: ${report.timestamp.toISOString()}`,
      });
      return score;
    }
    
    // Check if report is too old
    const hoursOld = (now - reportTime) / (1000 * 60 * 60);
    const pattern = DISASTER_PATTERNS[report.disasterType.toLowerCase()];
    
    if (pattern && hoursOld > pattern.typicalDuration * 2) {
      score *= 0.5;
      flags.push({
        type: 'time_anomaly',
        severity: 'medium',
        description: 'Report is significantly delayed',
        evidence: `Report is ${Math.round(hoursOld)} hours old`,
      });
    }
    
    return score;
  }

  /**
   * Analyze report content using NLP techniques
   */
  private analyzeContent(report: ReportData, flags: ValidationFlag[]): number {
    let score = 1.0;
    const description = report.description.toLowerCase();
    
    // Check for spam patterns
    for (const pattern of VALIDATION_MODEL.spamPatterns) {
      if (pattern.test(report.description)) {
        score *= 0.3;
        flags.push({
          type: 'spam',
          severity: 'high',
          description: 'Spam patterns detected',
          evidence: 'Description contains suspicious patterns',
        });
        break;
      }
    }
    
    // Check for disaster-specific keywords
    const pattern = DISASTER_PATTERNS[report.disasterType.toLowerCase()];
    if (pattern) {
      const keywordMatches = pattern.keywords.filter(kw => description.includes(kw.toLowerCase()));
      const keywordRatio = keywordMatches.length / pattern.keywords.length;
      
      if (keywordRatio < 0.1) {
        score *= 0.7;
        flags.push({
          type: 'inconsistent',
          severity: 'medium',
          description: 'Description lacks relevant keywords',
          evidence: `Only ${keywordMatches.length}/${pattern.keywords.length} expected keywords found`,
        });
      } else if (keywordRatio > 0.3) {
        score = Math.min(1.0, score * 1.1);
      }
    }
    
    // Check description length
    if (report.description.length < 20) {
      score *= 0.7;
      flags.push({
        type: 'inconsistent',
        severity: 'low',
        description: 'Description is too short',
        evidence: `Only ${report.description.length} characters`,
      });
    }
    
    // Sentiment analysis for severity consistency
    const detectedSeverity = this.detectSeverityFromText(description);
    if (detectedSeverity !== report.severity) {
      score *= 0.9;
      flags.push({
        type: 'inconsistent',
        severity: 'low',
        description: 'Severity mismatch with description',
        evidence: `Reported: ${report.severity}, Detected: ${detectedSeverity}`,
      });
    }
    
    return score;
  }

  /**
   * Cross-reference with other reports
   */
  private crossReferenceReports(
    report: ReportData,
    crossReferences: CrossReference[],
    flags: ValidationFlag[]
  ): number {
    let score = 0.5; // Start neutral without cross-references
    let supportingReports = 0;
    let conflictingReports = 0;
    
    const recentReports = Array.from(REPORT_STORAGE.values()).filter(r => {
      const timeDiff = Math.abs(report.timestamp.getTime() - r.timestamp.getTime());
      return timeDiff < 24 * 60 * 60 * 1000 && r.id !== report.id; // Within 24 hours
    });
    
    for (const otherReport of recentReports) {
      const distance = this.calculateDistance(
        report.location.latitude,
        report.location.longitude,
        otherReport.location.latitude,
        otherReport.location.longitude
      );
      
      const pattern = DISASTER_PATTERNS[report.disasterType.toLowerCase()];
      const maxDistance = pattern?.spreadRadius || 50;
      
      if (distance <= maxDistance) {
        const similarity = this.calculateSimilarity(report, otherReport);
        const timeDifference = Math.abs(report.timestamp.getTime() - otherReport.timestamp.getTime()) / (1000 * 60);
        
        const supports = similarity > 0.5 && otherReport.disasterType === report.disasterType;
        
        crossReferences.push({
          reportId: otherReport.id,
          similarity,
          distance,
          timeDifference,
          userId: otherReport.userId,
          supports,
        });
        
        if (supports) {
          supportingReports++;
        } else if (otherReport.disasterType !== report.disasterType) {
          conflictingReports++;
        }
      }
    }
    
    // Calculate score based on cross-references
    if (crossReferences.length > 0) {
      const supportRatio = supportingReports / crossReferences.length;
      score = 0.3 + (supportRatio * 0.7);
      
      if (conflictingReports > supportingReports) {
        flags.push({
          type: 'inconsistent',
          severity: 'medium',
          description: 'Conflicting reports in the area',
          evidence: `${conflictingReports} conflicting vs ${supportingReports} supporting`,
        });
      }
    }
    
    // Check for duplicates
    const duplicates = crossReferences.filter(cr => cr.similarity > 0.9 && cr.distance < 1);
    if (duplicates.length > 0) {
      score *= 0.5;
      flags.push({
        type: 'duplicate',
        severity: 'high',
        description: 'Potential duplicate report',
        evidence: `${duplicates.length} very similar report(s) found`,
      });
    }
    
    return score;
  }

  /**
   * Analyze images for authenticity
   */
  private async analyzeImages(images: string[], flags: ValidationFlag[]): Promise<number> {
    let score = 1.0;
    
    // Simulated image analysis
    for (const image of images) {
      // Check for known fake image patterns
      const isSuspicious = this.detectSuspiciousImage(image);
      
      if (isSuspicious) {
        score *= 0.5;
        flags.push({
          type: 'fake_image',
          severity: 'high',
          description: 'Suspicious image detected',
          evidence: 'Image may be manipulated or from an unrelated event',
        });
        break;
      }
    }
    
    return score;
  }

  /**
   * Calculate overall credibility score
   */
  private calculateCredibilityScore(scores: {
    reputationScore: number;
    locationScore: number;
    temporalScore: number;
    contentScore: number;
    crossRefScore: number;
    imageScore: number;
  }): number {
    const weighted = 
      scores.reputationScore * VALIDATION_MODEL.userReputationWeight +
      scores.locationScore * VALIDATION_MODEL.locationConsistencyWeight +
      scores.temporalScore * VALIDATION_MODEL.temporalConsistencyWeight +
      scores.contentScore * VALIDATION_MODEL.contentAnalysisWeight +
      scores.crossRefScore * VALIDATION_MODEL.crossReferenceWeight;
    
    // Apply image score as multiplier
    return Math.max(0, Math.min(1, weighted * scores.imageScore));
  }

  /**
   * Determine verification status based on score and flags
   */
  private determineVerificationStatus(
    score: number,
    flags: ValidationFlag[]
  ): 'pending' | 'verified' | 'disputed' | 'rejected' {
    const highSeverityFlags = flags.filter(f => f.severity === 'high').length;
    
    if (score < VALIDATION_MODEL.thresholds.reject || highSeverityFlags >= 3) {
      return 'rejected';
    }
    
    if (score >= VALIDATION_MODEL.thresholds.highCredibility && highSeverityFlags === 0) {
      return 'verified';
    }
    
    if (score < VALIDATION_MODEL.thresholds.mediumCredibility || highSeverityFlags >= 1) {
      return 'disputed';
    }
    
    return 'pending';
  }

  /**
   * Calculate confidence in the validation
   */
  private calculateConfidence(flags: ValidationFlag[], crossReferences: CrossReference[]): number {
    let confidence = 0.7; // Base confidence
    
    // More cross-references increase confidence
    confidence += Math.min(0.2, crossReferences.length * 0.02);
    
    // Flags reduce confidence
    confidence -= flags.length * 0.05;
    
    return Math.max(0.1, Math.min(1.0, confidence));
  }

  /**
   * Generate recommendations based on validation
   */
  private generateRecommendations(
    score: number,
    flags: ValidationFlag[],
    reputation: UserReputation
  ): string[] {
    const recommendations: string[] = [];
    
    if (score < VALIDATION_MODEL.thresholds.mediumCredibility) {
      recommendations.push('Manual review recommended before broadcasting');
    }
    
    if (flags.some(f => f.type === 'location_mismatch')) {
      recommendations.push('Request GPS coordinates verification from user');
    }
    
    if (flags.some(f => f.type === 'fake_image')) {
      recommendations.push('Request additional photos from different angles');
    }
    
    if (reputation.totalReports < 3) {
      recommendations.push('New user - consider additional verification');
    }
    
    if (score >= VALIDATION_MODEL.thresholds.highCredibility) {
      recommendations.push('Report can be auto-approved for immediate broadcast');
    }
    
    return recommendations;
  }

  /**
   * Helper: Get or create user reputation
   */
  private getUserReputation(userId: string): UserReputation {
    if (!USER_REPUTATIONS.has(userId)) {
      USER_REPUTATIONS.set(userId, {
        userId,
        totalReports: 0,
        verifiedReports: 0,
        flaggedReports: 0,
        averageAccuracy: 0.5,
        trustScore: 0.5,
        registrationDate: new Date(),
        lastActiveDate: new Date(),
        badges: [],
      });
    }
    return USER_REPUTATIONS.get(userId)!;
  }

  /**
   * Helper: Update user reputation
   */
  private updateUserReputation(userId: string, validationScore: number): void {
    const reputation = this.getUserReputation(userId);
    reputation.totalReports++;
    reputation.lastActiveDate = new Date();
    
    // Update running average accuracy
    reputation.averageAccuracy = 
      (reputation.averageAccuracy * (reputation.totalReports - 1) + validationScore) / 
      reputation.totalReports;
    
    // Update trust score
    if (validationScore >= VALIDATION_MODEL.thresholds.highCredibility) {
      reputation.verifiedReports++;
      reputation.trustScore = Math.min(1.0, reputation.trustScore + 0.05);
    } else if (validationScore < VALIDATION_MODEL.thresholds.lowCredibility) {
      reputation.flaggedReports++;
      reputation.trustScore = Math.max(0, reputation.trustScore - 0.1);
    }
    
    // Award badges
    if (reputation.verifiedReports >= 10 && !reputation.badges.includes('trusted_reporter')) {
      reputation.badges.push('trusted_reporter');
    }
    if (reputation.verifiedReports >= 50 && !reputation.badges.includes('veteran_reporter')) {
      reputation.badges.push('veteran_reporter');
    }
    
    USER_REPUTATIONS.set(userId, reputation);
  }

  /**
   * Helper: Calculate distance between coordinates (Haversine formula)
   */
  private calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const R = 6371; // Earth's radius in km
    const dLat = this.toRad(lat2 - lat1);
    const dLon = this.toRad(lon2 - lon1);
    const a = 
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.toRad(lat1)) * Math.cos(this.toRad(lat2)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  private toRad(deg: number): number {
    return deg * (Math.PI / 180);
  }

  /**
   * Helper: Calculate similarity between reports
   */
  private calculateSimilarity(report1: ReportData, report2: ReportData): number {
    let similarity = 0;
    
    // Same disaster type
    if (report1.disasterType === report2.disasterType) {
      similarity += 0.4;
    }
    
    // Similar severity
    const severityLevels = { low: 1, medium: 2, high: 3, critical: 4 };
    const severityDiff = Math.abs(severityLevels[report1.severity] - severityLevels[report2.severity]);
    similarity += (4 - severityDiff) / 4 * 0.2;
    
    // Text similarity (simple word overlap)
    const words1 = new Set(report1.description.toLowerCase().split(/\s+/));
    const words2 = new Set(report2.description.toLowerCase().split(/\s+/));
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    similarity += (intersection.size / union.size) * 0.4;
    
    return similarity;
  }

  /**
   * Helper: Validate address consistency
   */
  private validateAddressConsistency(lat: number, lng: number, address: string): boolean {
    // Simulated address validation
    // In production, would use geocoding API
    return true;
  }

  /**
   * Helper: Check if location is in populated area
   */
  private checkPopulatedArea(lat: number, lng: number): boolean {
    // Simulated population check
    // In production, would use population density API
    return Math.random() > 0.2;
  }

  /**
   * Helper: Detect severity from text
   */
  private detectSeverityFromText(text: string): 'low' | 'medium' | 'high' | 'critical' {
    const criticalWords = ['catastrophic', 'devastating', 'massive', 'emergency', 'critical', 'urgent'];
    const highWords = ['severe', 'major', 'significant', 'dangerous', 'serious'];
    const mediumWords = ['moderate', 'substantial', 'notable', 'concerning'];
    
    text = text.toLowerCase();
    
    if (criticalWords.some(w => text.includes(w))) return 'critical';
    if (highWords.some(w => text.includes(w))) return 'high';
    if (mediumWords.some(w => text.includes(w))) return 'medium';
    return 'low';
  }

  /**
   * Helper: Detect suspicious images
   */
  private detectSuspiciousImage(imageUrl: string): boolean {
    // Simulated image analysis
    // In production, would use ML image forensics
    return Math.random() < 0.05; // 5% chance of suspicious image
  }

  /**
   * Initialize sample data
   */
  private initializeSampleData(): void {
    // Sample user reputations
    const sampleUsers = [
      { userId: 'user_001', verified: 25, flagged: 1, trust: 0.9 },
      { userId: 'user_002', verified: 10, flagged: 3, trust: 0.7 },
      { userId: 'user_003', verified: 2, flagged: 0, trust: 0.5 },
    ];
    
    sampleUsers.forEach(u => {
      USER_REPUTATIONS.set(u.userId, {
        userId: u.userId,
        totalReports: u.verified + u.flagged,
        verifiedReports: u.verified,
        flaggedReports: u.flagged,
        averageAccuracy: u.trust,
        trustScore: u.trust,
        registrationDate: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
        lastActiveDate: new Date(),
        badges: u.verified >= 10 ? ['trusted_reporter'] : [],
      });
    });
  }

  /**
   * Batch validation for multiple reports
   */
  public async validateBatch(reports: ReportData[]): Promise<ValidationResult[]> {
    const results: ValidationResult[] = [];
    
    for (const report of reports) {
      const result = await this.validateReport(report);
      results.push(result);
    }
    
    return results;
  }

  /**
   * Get user reputation by ID
   */
  public getReputationById(userId: string): UserReputation | undefined {
    return USER_REPUTATIONS.get(userId);
  }

  /**
   * Get all reports for analysis
   */
  public getAllReports(): ReportData[] {
    return Array.from(REPORT_STORAGE.values());
  }
}

export const dataValidationService = DataValidationService.getInstance();
export type { ReportData, ValidationResult, UserReputation, ValidationFlag, CrossReference };
