/**
 * Volunteer Coordination Service
 * Manage volunteer registration, assignments, and activities
 */

// Volunteer status
type VolunteerStatus = 'pending' | 'approved' | 'active' | 'inactive' | 'suspended' | 'retired';

// Availability status
type AvailabilityStatus = 'available' | 'busy' | 'unavailable' | 'on_assignment';

// Skill category
type SkillCategory = 
  | 'medical'
  | 'rescue'
  | 'logistics'
  | 'communication'
  | 'transportation'
  | 'construction'
  | 'counseling'
  | 'administration'
  | 'cooking'
  | 'childcare'
  | 'translation'
  | 'technology'
  | 'security'
  | 'first_aid';

// Assignment status
type AssignmentStatus = 'pending' | 'assigned' | 'in_progress' | 'completed' | 'cancelled';

// Assignment priority
type AssignmentPriority = 'critical' | 'high' | 'medium' | 'low';

// Team type
type TeamType = 'search_rescue' | 'medical' | 'logistics' | 'evacuation' | 'relief' | 'recovery' | 'communication';

// Volunteer profile
interface VolunteerProfile {
  id: string;
  userId: string;
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  alternatePhone?: string;
  dateOfBirth: Date;
  gender: 'male' | 'female' | 'other' | 'prefer_not_say';
  bloodGroup?: string;
  address: VolunteerAddress;
  status: VolunteerStatus;
  availabilityStatus: AvailabilityStatus;
  skills: VolunteerSkill[];
  certifications: Certification[];
  languages: Language[];
  emergencyContact: EmergencyContact;
  equipment: OwnedEquipment[];
  vehicleInfo?: VehicleInfo;
  preferences: VolunteerPreferences;
  stats: VolunteerStats;
  registeredAt: Date;
  approvedAt?: Date;
  lastActiveAt: Date;
  rating: number;
  badges: VolunteerBadge[];
  notes?: string;
}

// Volunteer address
interface VolunteerAddress {
  street: string;
  city: string;
  district: string;
  state: string;
  pincode: string;
  coordinates?: { lat: number; lng: number };
}

// Volunteer skill
interface VolunteerSkill {
  category: SkillCategory;
  name: string;
  level: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  yearsExperience: number;
  isVerified: boolean;
  verifiedBy?: string;
  verifiedAt?: Date;
}

// Certification
interface Certification {
  id: string;
  name: string;
  issuingAuthority: string;
  issueDate: Date;
  expiryDate?: Date;
  certificateNumber?: string;
  documentUrl?: string;
  isVerified: boolean;
}

// Language
interface Language {
  code: string;
  name: string;
  proficiency: 'basic' | 'conversational' | 'fluent' | 'native';
  canRead: boolean;
  canWrite: boolean;
}

// Emergency contact
interface EmergencyContact {
  name: string;
  relationship: string;
  phone: string;
  alternatePhone?: string;
}

// Owned equipment
interface OwnedEquipment {
  type: string;
  description: string;
  quantity: number;
  isAvailable: boolean;
}

// Vehicle info
interface VehicleInfo {
  type: 'two_wheeler' | 'four_wheeler' | 'truck' | 'boat' | 'other';
  make?: string;
  model?: string;
  registrationNumber: string;
  capacity?: number;
  isAvailable: boolean;
}

// Volunteer preferences
interface VolunteerPreferences {
  maxTravelDistance: number; // in km
  preferredAreas: string[];
  preferredDisasterTypes: string[];
  availableDays: ('monday' | 'tuesday' | 'wednesday' | 'thursday' | 'friday' | 'saturday' | 'sunday')[];
  availableHours: { start: string; end: string };
  willingToTravel: boolean;
  canWorkNights: boolean;
  canWorkWeekends: boolean;
  physicalLimitations?: string;
}

// Volunteer stats
interface VolunteerStats {
  totalAssignments: number;
  completedAssignments: number;
  totalHours: number;
  trainingsCompleted: number;
  peopleHelped: number;
  reputationScore: number;
}

// Volunteer badge
interface VolunteerBadge {
  id: string;
  name: string;
  icon: string;
  description: string;
  earnedAt: Date;
  category: 'achievement' | 'skill' | 'service' | 'special';
}

// Assignment
interface Assignment {
  id: string;
  title: string;
  description: string;
  type: TeamType;
  location: AssignmentLocation;
  disasterId?: string;
  priority: AssignmentPriority;
  status: AssignmentStatus;
  requiredSkills: SkillCategory[];
  requiredVolunteers: number;
  assignedVolunteers: AssignedVolunteer[];
  startTime: Date;
  endTime?: Date;
  estimatedDuration: number; // in hours
  instructions: string[];
  equipment: string[];
  coordinator: CoordinatorInfo;
  checkpoints: Checkpoint[];
  tasks: AssignmentTask[];
  createdAt: Date;
  updatedAt: Date;
  completedAt?: Date;
  notes?: string;
}

// Assignment location
interface AssignmentLocation {
  name: string;
  address: string;
  coordinates: { lat: number; lng: number };
  district: string;
  state: string;
  meetingPoint?: string;
}

// Assigned volunteer
interface AssignedVolunteer {
  volunteerId: string;
  volunteerName: string;
  assignedAt: Date;
  confirmedAt?: Date;
  checkInTime?: Date;
  checkOutTime?: Date;
  status: 'pending' | 'confirmed' | 'checked_in' | 'checked_out' | 'no_show';
  role?: string;
  feedback?: VolunteerFeedback;
}

// Coordinator info
interface CoordinatorInfo {
  id: string;
  name: string;
  phone: string;
  email: string;
}

// Checkpoint
interface Checkpoint {
  id: string;
  title: string;
  description?: string;
  dueTime?: Date;
  completedAt?: Date;
  completedBy?: string;
}

// Assignment task
interface AssignmentTask {
  id: string;
  title: string;
  description?: string;
  assignedTo?: string;
  status: 'pending' | 'in_progress' | 'completed';
  priority: AssignmentPriority;
  completedAt?: Date;
}

// Volunteer feedback
interface VolunteerFeedback {
  rating: number;
  comment?: string;
  badges?: string[];
  givenBy: string;
  givenAt: Date;
}

// Team
interface VolunteerTeam {
  id: string;
  name: string;
  type: TeamType;
  description: string;
  leaderId: string;
  leaderName: string;
  members: TeamMember[];
  maxMembers: number;
  district: string;
  state: string;
  isActive: boolean;
  createdAt: Date;
  assignments: string[];
  specializations: SkillCategory[];
}

// Team member
interface TeamMember {
  volunteerId: string;
  name: string;
  role: 'leader' | 'deputy' | 'member';
  joinedAt: Date;
  skills: SkillCategory[];
}

// Training
interface Training {
  id: string;
  title: string;
  description: string;
  type: 'online' | 'offline' | 'hybrid';
  category: SkillCategory;
  instructor: string;
  location?: string;
  startDate: Date;
  endDate: Date;
  duration: number; // in hours
  maxParticipants: number;
  enrolledParticipants: string[];
  completedParticipants: string[];
  prerequisites: string[];
  certificateProvided: boolean;
  status: 'scheduled' | 'in_progress' | 'completed' | 'cancelled';
}

// Search filters
interface VolunteerSearchFilters {
  query?: string;
  skills?: SkillCategory[];
  status?: VolunteerStatus[];
  availability?: AvailabilityStatus[];
  location?: { state?: string; district?: string };
  hasVehicle?: boolean;
  certifications?: string[];
  languages?: string[];
  minRating?: number;
  maxDistance?: number;
  coordinates?: { lat: number; lng: number };
}

// Dashboard stats
interface VolunteerDashboardStats {
  totalVolunteers: number;
  activeVolunteers: number;
  availableVolunteers: number;
  onAssignment: number;
  pendingApprovals: number;
  totalTeams: number;
  activeAssignments: number;
  completedAssignments: number;
  totalHoursContributed: number;
  bySkill: { skill: SkillCategory; count: number }[];
  byState: { state: string; count: number }[];
}

// Indian states for sample data
const INDIAN_STATES = [
  'Kerala', 'Tamil Nadu', 'Karnataka', 'Maharashtra', 'Gujarat',
  'Rajasthan', 'Delhi', 'Uttar Pradesh', 'West Bengal', 'Odisha',
];

// Available badges
const VOLUNTEER_BADGES: Omit<VolunteerBadge, 'earnedAt'>[] = [
  { id: 'first_mission', name: 'First Mission', icon: '🎯', description: 'Completed first assignment', category: 'achievement' },
  { id: 'helping_hand', name: 'Helping Hand', icon: '🤝', description: 'Helped 50+ people', category: 'service' },
  { id: 'night_owl', name: 'Night Owl', icon: '🦉', description: 'Completed 5 night assignments', category: 'achievement' },
  { id: 'first_responder', name: 'First Responder', icon: '🚨', description: 'First to respond to an emergency', category: 'achievement' },
  { id: 'medical_hero', name: 'Medical Hero', icon: '⚕️', description: 'Provided critical medical assistance', category: 'skill' },
  { id: 'logistics_master', name: 'Logistics Master', icon: '📦', description: 'Coordinated major logistics operation', category: 'skill' },
  { id: 'community_champion', name: 'Community Champion', icon: '🏆', description: '100+ hours of service', category: 'service' },
  { id: 'trainer', name: 'Trainer', icon: '🎓', description: 'Trained other volunteers', category: 'special' },
  { id: 'team_player', name: 'Team Player', icon: '⭐', description: 'Outstanding team collaboration', category: 'achievement' },
  { id: 'quick_responder', name: 'Quick Responder', icon: '⚡', description: 'Responded within 30 minutes', category: 'achievement' },
];

class VolunteerCoordinationService {
  private static instance: VolunteerCoordinationService;
  private volunteers: Map<string, VolunteerProfile> = new Map();
  private assignments: Map<string, Assignment> = new Map();
  private teams: Map<string, VolunteerTeam> = new Map();
  private trainings: Map<string, Training> = new Map();
  private listeners: ((event: string, data: unknown) => void)[] = [];

  private constructor() {
    this.initializeSampleData();
  }

  public static getInstance(): VolunteerCoordinationService {
    if (!VolunteerCoordinationService.instance) {
      VolunteerCoordinationService.instance = new VolunteerCoordinationService();
    }
    return VolunteerCoordinationService.instance;
  }

  /**
   * Initialize sample data
   */
  private initializeSampleData(): void {
    // Create 50 sample volunteers
    const skills: SkillCategory[] = ['medical', 'rescue', 'logistics', 'communication', 'transportation', 'first_aid'];
    
    for (let i = 1; i <= 50; i++) {
      const stateIndex = i % INDIAN_STATES.length;
      const volunteer: VolunteerProfile = {
        id: `vol-${i.toString().padStart(4, '0')}`,
        userId: `user-${i}`,
        firstName: `Volunteer`,
        lastName: `${i}`,
        email: `volunteer${i}@example.com`,
        phone: `+91${9000000000 + i}`,
        dateOfBirth: new Date(1985 + (i % 20), i % 12, (i % 28) + 1),
        gender: i % 4 === 0 ? 'female' : 'male',
        bloodGroup: ['A+', 'B+', 'O+', 'AB+', 'A-', 'B-', 'O-', 'AB-'][i % 8],
        address: {
          street: `${i} Main Street`,
          city: 'City',
          district: 'District',
          state: INDIAN_STATES[stateIndex],
          pincode: `${600000 + i}`,
        },
        status: i % 10 === 0 ? 'pending' : 'active',
        availabilityStatus: i % 5 === 0 ? 'busy' : 'available',
        skills: [
          { category: skills[i % skills.length], name: skills[i % skills.length], level: 'intermediate', yearsExperience: 2, isVerified: true },
          { category: skills[(i + 1) % skills.length], name: skills[(i + 1) % skills.length], level: 'beginner', yearsExperience: 1, isVerified: false },
        ],
        certifications: [
          { id: `cert-${i}`, name: 'First Aid Certification', issuingAuthority: 'Red Cross', issueDate: new Date(), isVerified: true },
        ],
        languages: [
          { code: 'en', name: 'English', proficiency: 'fluent', canRead: true, canWrite: true },
          { code: 'hi', name: 'Hindi', proficiency: 'native', canRead: true, canWrite: true },
        ],
        emergencyContact: {
          name: 'Emergency Contact',
          relationship: 'family',
          phone: `+91${9100000000 + i}`,
        },
        equipment: [],
        vehicleInfo: i % 3 === 0 ? {
          type: 'two_wheeler',
          registrationNumber: `KA-${i}-XX-${1000 + i}`,
          isAvailable: true,
        } : undefined,
        preferences: {
          maxTravelDistance: 50,
          preferredAreas: [INDIAN_STATES[stateIndex]],
          preferredDisasterTypes: ['flood', 'earthquake'],
          availableDays: ['saturday', 'sunday'],
          availableHours: { start: '09:00', end: '18:00' },
          willingToTravel: true,
          canWorkNights: i % 2 === 0,
          canWorkWeekends: true,
        },
        stats: {
          totalAssignments: Math.floor(Math.random() * 20),
          completedAssignments: Math.floor(Math.random() * 15),
          totalHours: Math.floor(Math.random() * 200),
          trainingsCompleted: Math.floor(Math.random() * 5),
          peopleHelped: Math.floor(Math.random() * 100),
          reputationScore: Math.floor(Math.random() * 1000),
        },
        registeredAt: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000),
        approvedAt: new Date(),
        lastActiveAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000),
        rating: 3.5 + Math.random() * 1.5,
        badges: [VOLUNTEER_BADGES[i % VOLUNTEER_BADGES.length] as VolunteerBadge].map(b => ({ ...b, earnedAt: new Date() })),
      };
      this.volunteers.set(volunteer.id, volunteer);
    }

    // Create sample teams
    const teamTypes: TeamType[] = ['search_rescue', 'medical', 'logistics', 'evacuation', 'relief'];
    for (let i = 1; i <= 10; i++) {
      const team: VolunteerTeam = {
        id: `team-${i.toString().padStart(3, '0')}`,
        name: `${teamTypes[(i - 1) % teamTypes.length].replace('_', ' ').toUpperCase()} Team ${i}`,
        type: teamTypes[(i - 1) % teamTypes.length],
        description: `Regional ${teamTypes[(i - 1) % teamTypes.length]} team`,
        leaderId: `vol-${i.toString().padStart(4, '0')}`,
        leaderName: `Volunteer ${i}`,
        members: [],
        maxMembers: 15,
        district: 'District',
        state: INDIAN_STATES[i % INDIAN_STATES.length],
        isActive: true,
        createdAt: new Date(),
        assignments: [],
        specializations: [skills[i % skills.length]],
      };
      this.teams.set(team.id, team);
    }

    // Create sample assignments
    for (let i = 1; i <= 20; i++) {
      const assignment: Assignment = {
        id: `assign-${i.toString().padStart(4, '0')}`,
        title: `Assignment ${i}`,
        description: `Emergency response assignment for disaster relief`,
        type: teamTypes[(i - 1) % teamTypes.length],
        location: {
          name: `Location ${i}`,
          address: `Address ${i}`,
          coordinates: { lat: 10 + Math.random() * 10, lng: 75 + Math.random() * 10 },
          district: 'District',
          state: INDIAN_STATES[i % INDIAN_STATES.length],
        },
        priority: (['critical', 'high', 'medium', 'low'] as AssignmentPriority[])[i % 4],
        status: (['pending', 'assigned', 'in_progress', 'completed'] as AssignmentStatus[])[i % 4],
        requiredSkills: [skills[i % skills.length]],
        requiredVolunteers: 5 + (i % 10),
        assignedVolunteers: [],
        startTime: new Date(Date.now() + i * 24 * 60 * 60 * 1000),
        estimatedDuration: 4 + (i % 8),
        instructions: ['Report to coordinator', 'Bring ID proof', 'Follow safety protocols'],
        equipment: ['First aid kit', 'Communication device'],
        coordinator: {
          id: 'coord-1',
          name: 'District Coordinator',
          phone: '+919876543210',
          email: 'coordinator@example.com',
        },
        checkpoints: [],
        tasks: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      this.assignments.set(assignment.id, assignment);
    }
  }

  /**
   * Register volunteer
   */
  public async registerVolunteer(data: Omit<VolunteerProfile, 'id' | 'status' | 'stats' | 'registeredAt' | 'lastActiveAt' | 'rating' | 'badges'>): Promise<VolunteerProfile> {
    const id = `vol-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`;
    const volunteer: VolunteerProfile = {
      ...data,
      id,
      status: 'pending',
      stats: {
        totalAssignments: 0,
        completedAssignments: 0,
        totalHours: 0,
        trainingsCompleted: 0,
        peopleHelped: 0,
        reputationScore: 0,
      },
      registeredAt: new Date(),
      lastActiveAt: new Date(),
      rating: 0,
      badges: [],
    };

    this.volunteers.set(id, volunteer);
    this.emit('volunteer_registered', volunteer);
    return volunteer;
  }

  /**
   * Get volunteer
   */
  public getVolunteer(volunteerId: string): VolunteerProfile | undefined {
    return this.volunteers.get(volunteerId);
  }

  /**
   * Update volunteer
   */
  public updateVolunteer(volunteerId: string, updates: Partial<VolunteerProfile>): VolunteerProfile | null {
    const volunteer = this.volunteers.get(volunteerId);
    if (!volunteer) return null;

    Object.assign(volunteer, updates);
    volunteer.lastActiveAt = new Date();
    this.emit('volunteer_updated', volunteer);
    return volunteer;
  }

  /**
   * Approve volunteer
   */
  public approveVolunteer(volunteerId: string, approvedBy: string): boolean {
    const volunteer = this.volunteers.get(volunteerId);
    if (!volunteer || volunteer.status !== 'pending') return false;

    volunteer.status = 'active';
    volunteer.approvedAt = new Date();
    this.emit('volunteer_approved', volunteer);
    return true;
  }

  /**
   * Search volunteers
   */
  public async searchVolunteers(filters: VolunteerSearchFilters, page: number = 1, pageSize: number = 20): Promise<{ volunteers: VolunteerProfile[]; total: number }> {
    let results = Array.from(this.volunteers.values());

    if (filters.query) {
      const query = filters.query.toLowerCase();
      results = results.filter((v) =>
        v.firstName.toLowerCase().includes(query) ||
        v.lastName.toLowerCase().includes(query) ||
        v.email.toLowerCase().includes(query)
      );
    }

    if (filters.skills?.length) {
      results = results.filter((v) => v.skills.some((s) => filters.skills!.includes(s.category)));
    }

    if (filters.status?.length) {
      results = results.filter((v) => filters.status!.includes(v.status));
    }

    if (filters.availability?.length) {
      results = results.filter((v) => filters.availability!.includes(v.availabilityStatus));
    }

    if (filters.location?.state) {
      results = results.filter((v) => v.address.state === filters.location!.state);
    }

    if (filters.location?.district) {
      results = results.filter((v) => v.address.district === filters.location!.district);
    }

    if (filters.hasVehicle) {
      results = results.filter((v) => !!v.vehicleInfo?.isAvailable);
    }

    if (filters.minRating) {
      results = results.filter((v) => v.rating >= filters.minRating!);
    }

    if (filters.languages?.length) {
      results = results.filter((v) => v.languages.some((l) => filters.languages!.includes(l.code)));
    }

    const total = results.length;
    const startIndex = (page - 1) * pageSize;
    const paginatedResults = results.slice(startIndex, startIndex + pageSize);

    return { volunteers: paginatedResults, total };
  }

  /**
   * Get available volunteers
   */
  public getAvailableVolunteers(skills?: SkillCategory[], location?: { state?: string; district?: string }): VolunteerProfile[] {
    return Array.from(this.volunteers.values()).filter((v) => {
      if (v.status !== 'active' || v.availabilityStatus !== 'available') return false;
      if (skills?.length && !v.skills.some((s) => skills.includes(s.category))) return false;
      if (location?.state && v.address.state !== location.state) return false;
      if (location?.district && v.address.district !== location.district) return false;
      return true;
    });
  }

  /**
   * Update availability
   */
  public updateAvailability(volunteerId: string, status: AvailabilityStatus): boolean {
    const volunteer = this.volunteers.get(volunteerId);
    if (!volunteer) return false;

    volunteer.availabilityStatus = status;
    volunteer.lastActiveAt = new Date();
    this.emit('availability_updated', { volunteerId, status });
    return true;
  }

  /**
   * Create assignment
   */
  public async createAssignment(data: Omit<Assignment, 'id' | 'status' | 'assignedVolunteers' | 'createdAt' | 'updatedAt'>): Promise<Assignment> {
    const id = `assign-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`;
    const assignment: Assignment = {
      ...data,
      id,
      status: 'pending',
      assignedVolunteers: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.assignments.set(id, assignment);
    this.emit('assignment_created', assignment);
    return assignment;
  }

  /**
   * Get assignment
   */
  public getAssignment(assignmentId: string): Assignment | undefined {
    return this.assignments.get(assignmentId);
  }

  /**
   * Assign volunteer
   */
  public assignVolunteer(assignmentId: string, volunteerId: string, role?: string): boolean {
    const assignment = this.assignments.get(assignmentId);
    const volunteer = this.volunteers.get(volunteerId);

    if (!assignment || !volunteer) return false;
    if (assignment.assignedVolunteers.some((v) => v.volunteerId === volunteerId)) return false;

    assignment.assignedVolunteers.push({
      volunteerId,
      volunteerName: `${volunteer.firstName} ${volunteer.lastName}`,
      assignedAt: new Date(),
      status: 'pending',
      role,
    });

    if (assignment.status === 'pending') {
      assignment.status = 'assigned';
    }

    volunteer.availabilityStatus = 'on_assignment';
    assignment.updatedAt = new Date();

    this.emit('volunteer_assigned', { assignmentId, volunteerId });
    return true;
  }

  /**
   * Confirm assignment
   */
  public confirmAssignment(assignmentId: string, volunteerId: string): boolean {
    const assignment = this.assignments.get(assignmentId);
    if (!assignment) return false;

    const assignedVolunteer = assignment.assignedVolunteers.find((v) => v.volunteerId === volunteerId);
    if (!assignedVolunteer) return false;

    assignedVolunteer.status = 'confirmed';
    assignedVolunteer.confirmedAt = new Date();
    assignment.updatedAt = new Date();

    this.emit('assignment_confirmed', { assignmentId, volunteerId });
    return true;
  }

  /**
   * Check in volunteer
   */
  public checkIn(assignmentId: string, volunteerId: string): boolean {
    const assignment = this.assignments.get(assignmentId);
    if (!assignment) return false;

    const assignedVolunteer = assignment.assignedVolunteers.find((v) => v.volunteerId === volunteerId);
    if (!assignedVolunteer || assignedVolunteer.status !== 'confirmed') return false;

    assignedVolunteer.status = 'checked_in';
    assignedVolunteer.checkInTime = new Date();

    if (assignment.status === 'assigned') {
      assignment.status = 'in_progress';
    }

    assignment.updatedAt = new Date();
    this.emit('volunteer_checked_in', { assignmentId, volunteerId });
    return true;
  }

  /**
   * Check out volunteer
   */
  public checkOut(assignmentId: string, volunteerId: string): boolean {
    const assignment = this.assignments.get(assignmentId);
    if (!assignment) return false;

    const assignedVolunteer = assignment.assignedVolunteers.find((v) => v.volunteerId === volunteerId);
    if (!assignedVolunteer || assignedVolunteer.status !== 'checked_in') return false;

    assignedVolunteer.status = 'checked_out';
    assignedVolunteer.checkOutTime = new Date();

    const volunteer = this.volunteers.get(volunteerId);
    if (volunteer) {
      volunteer.availabilityStatus = 'available';
      const hoursWorked = (assignedVolunteer.checkOutTime.getTime() - assignedVolunteer.checkInTime!.getTime()) / (1000 * 60 * 60);
      volunteer.stats.totalHours += hoursWorked;
      volunteer.stats.totalAssignments++;
    }

    assignment.updatedAt = new Date();
    this.emit('volunteer_checked_out', { assignmentId, volunteerId });
    return true;
  }

  /**
   * Complete assignment
   */
  public completeAssignment(assignmentId: string): boolean {
    const assignment = this.assignments.get(assignmentId);
    if (!assignment) return false;

    assignment.status = 'completed';
    assignment.completedAt = new Date();
    assignment.updatedAt = new Date();

    // Update volunteer stats
    assignment.assignedVolunteers.forEach((av) => {
      const volunteer = this.volunteers.get(av.volunteerId);
      if (volunteer) {
        volunteer.stats.completedAssignments++;
        volunteer.availabilityStatus = 'available';
      }
    });

    this.emit('assignment_completed', assignment);
    return true;
  }

  /**
   * Get active assignments
   */
  public getActiveAssignments(state?: string): Assignment[] {
    return Array.from(this.assignments.values()).filter((a) => {
      if (['completed', 'cancelled'].includes(a.status)) return false;
      if (state && a.location.state !== state) return false;
      return true;
    });
  }

  /**
   * Create team
   */
  public async createTeam(data: Omit<VolunteerTeam, 'id' | 'createdAt' | 'assignments'>): Promise<VolunteerTeam> {
    const id = `team-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`;
    const team: VolunteerTeam = {
      ...data,
      id,
      createdAt: new Date(),
      assignments: [],
    };

    this.teams.set(id, team);
    this.emit('team_created', team);
    return team;
  }

  /**
   * Get team
   */
  public getTeam(teamId: string): VolunteerTeam | undefined {
    return this.teams.get(teamId);
  }

  /**
   * Add team member
   */
  public addTeamMember(teamId: string, volunteerId: string, role: 'leader' | 'deputy' | 'member' = 'member'): boolean {
    const team = this.teams.get(teamId);
    const volunteer = this.volunteers.get(volunteerId);

    if (!team || !volunteer) return false;
    if (team.members.length >= team.maxMembers) return false;
    if (team.members.some((m) => m.volunteerId === volunteerId)) return false;

    team.members.push({
      volunteerId,
      name: `${volunteer.firstName} ${volunteer.lastName}`,
      role,
      joinedAt: new Date(),
      skills: volunteer.skills.map((s) => s.category),
    });

    this.emit('team_member_added', { teamId, volunteerId });
    return true;
  }

  /**
   * Remove team member
   */
  public removeTeamMember(teamId: string, volunteerId: string): boolean {
    const team = this.teams.get(teamId);
    if (!team) return false;

    const index = team.members.findIndex((m) => m.volunteerId === volunteerId);
    if (index === -1) return false;

    team.members.splice(index, 1);
    this.emit('team_member_removed', { teamId, volunteerId });
    return true;
  }

  /**
   * Get teams
   */
  public getTeams(filters?: { type?: TeamType; state?: string; isActive?: boolean }): VolunteerTeam[] {
    let teams = Array.from(this.teams.values());

    if (filters?.type) {
      teams = teams.filter((t) => t.type === filters.type);
    }

    if (filters?.state) {
      teams = teams.filter((t) => t.state === filters.state);
    }

    if (filters?.isActive !== undefined) {
      teams = teams.filter((t) => t.isActive === filters.isActive);
    }

    return teams;
  }

  /**
   * Create training
   */
  public async createTraining(data: Omit<Training, 'id' | 'enrolledParticipants' | 'completedParticipants' | 'status'>): Promise<Training> {
    const id = `training-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`;
    const training: Training = {
      ...data,
      id,
      enrolledParticipants: [],
      completedParticipants: [],
      status: 'scheduled',
    };

    this.trainings.set(id, training);
    this.emit('training_created', training);
    return training;
  }

  /**
   * Enroll in training
   */
  public enrollInTraining(trainingId: string, volunteerId: string): boolean {
    const training = this.trainings.get(trainingId);
    if (!training) return false;
    if (training.enrolledParticipants.length >= training.maxParticipants) return false;
    if (training.enrolledParticipants.includes(volunteerId)) return false;

    training.enrolledParticipants.push(volunteerId);
    this.emit('training_enrolled', { trainingId, volunteerId });
    return true;
  }

  /**
   * Complete training
   */
  public completeTraining(trainingId: string, volunteerId: string): boolean {
    const training = this.trainings.get(trainingId);
    const volunteer = this.volunteers.get(volunteerId);

    if (!training || !volunteer) return false;
    if (!training.enrolledParticipants.includes(volunteerId)) return false;
    if (training.completedParticipants.includes(volunteerId)) return false;

    training.completedParticipants.push(volunteerId);
    volunteer.stats.trainingsCompleted++;

    this.emit('training_completed', { trainingId, volunteerId });
    return true;
  }

  /**
   * Get dashboard stats
   */
  public getDashboardStats(): VolunteerDashboardStats {
    const volunteers = Array.from(this.volunteers.values());
    const assignments = Array.from(this.assignments.values());

    const skillCounts = new Map<SkillCategory, number>();
    const stateCounts = new Map<string, number>();

    volunteers.forEach((v) => {
      v.skills.forEach((s) => {
        skillCounts.set(s.category, (skillCounts.get(s.category) || 0) + 1);
      });
      stateCounts.set(v.address.state, (stateCounts.get(v.address.state) || 0) + 1);
    });

    return {
      totalVolunteers: volunteers.length,
      activeVolunteers: volunteers.filter((v) => v.status === 'active').length,
      availableVolunteers: volunteers.filter((v) => v.availabilityStatus === 'available').length,
      onAssignment: volunteers.filter((v) => v.availabilityStatus === 'on_assignment').length,
      pendingApprovals: volunteers.filter((v) => v.status === 'pending').length,
      totalTeams: this.teams.size,
      activeAssignments: assignments.filter((a) => ['assigned', 'in_progress'].includes(a.status)).length,
      completedAssignments: assignments.filter((a) => a.status === 'completed').length,
      totalHoursContributed: volunteers.reduce((sum, v) => sum + v.stats.totalHours, 0),
      bySkill: Array.from(skillCounts.entries()).map(([skill, count]) => ({ skill, count })),
      byState: Array.from(stateCounts.entries()).map(([state, count]) => ({ state, count })),
    };
  }

  /**
   * Award badge
   */
  public awardBadge(volunteerId: string, badgeId: string): boolean {
    const volunteer = this.volunteers.get(volunteerId);
    if (!volunteer) return false;

    const badgeTemplate = VOLUNTEER_BADGES.find((b) => b.id === badgeId);
    if (!badgeTemplate) return false;

    if (volunteer.badges.some((b) => b.id === badgeId)) return false;

    volunteer.badges.push({ ...badgeTemplate, earnedAt: new Date() });
    this.emit('badge_awarded', { volunteerId, badgeId });
    return true;
  }

  /**
   * Get volunteer history
   */
  public getVolunteerHistory(volunteerId: string): Assignment[] {
    return Array.from(this.assignments.values()).filter((a) =>
      a.assignedVolunteers.some((v) => v.volunteerId === volunteerId)
    );
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

export const volunteerCoordinationService = VolunteerCoordinationService.getInstance();
export type {
  VolunteerStatus,
  AvailabilityStatus,
  SkillCategory,
  AssignmentStatus,
  AssignmentPriority,
  TeamType,
  VolunteerProfile,
  VolunteerAddress,
  VolunteerSkill,
  Certification,
  Language,
  EmergencyContact,
  OwnedEquipment,
  VehicleInfo,
  VolunteerPreferences,
  VolunteerStats,
  VolunteerBadge,
  Assignment,
  AssignmentLocation,
  AssignedVolunteer,
  CoordinatorInfo,
  Checkpoint,
  AssignmentTask,
  VolunteerFeedback,
  VolunteerTeam,
  TeamMember,
  Training,
  VolunteerSearchFilters,
  VolunteerDashboardStats,
};
