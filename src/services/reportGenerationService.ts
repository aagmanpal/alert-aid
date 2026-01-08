/**
 * Report Generation Service
 * Automated disaster report generation and distribution
 */

// Report types
type ReportType = 
  | 'situation_report'
  | 'damage_assessment'
  | 'resource_status'
  | 'evacuation_summary'
  | 'shelter_status'
  | 'casualty_report'
  | 'relief_distribution'
  | 'financial_summary'
  | 'volunteer_activity'
  | 'incident_log'
  | 'weather_summary'
  | 'infrastructure_status'
  | 'communication_log'
  | 'daily_briefing'
  | 'weekly_summary'
  | 'monthly_analysis';

// Export formats
type ExportFormat = 'pdf' | 'excel' | 'csv' | 'word' | 'html' | 'json';

// Report status
type ReportStatus = 'draft' | 'pending_review' | 'approved' | 'published' | 'archived';

// Report priority
type ReportPriority = 'urgent' | 'high' | 'normal' | 'low';

// Report template
interface ReportTemplate {
  id: string;
  name: string;
  type: ReportType;
  description: string;
  sections: ReportSection[];
  defaultFormat: ExportFormat;
  logo?: string;
  headerText?: string;
  footerText?: string;
  styling: ReportStyling;
}

// Report section
interface ReportSection {
  id: string;
  title: string;
  type: 'text' | 'table' | 'chart' | 'image' | 'list' | 'metrics' | 'map';
  order: number;
  required: boolean;
  dataSource?: string;
  template?: string;
  config?: SectionConfig;
}

// Section configuration
interface SectionConfig {
  columns?: TableColumn[];
  chartType?: 'bar' | 'line' | 'pie' | 'area';
  showTotals?: boolean;
  maxItems?: number;
  groupBy?: string;
  sortBy?: string;
  filters?: Record<string, unknown>;
}

// Table column definition
interface TableColumn {
  key: string;
  header: string;
  width?: number;
  align?: 'left' | 'center' | 'right';
  format?: 'text' | 'number' | 'currency' | 'date' | 'percentage';
}

// Report styling
interface ReportStyling {
  fontFamily: string;
  fontSize: number;
  primaryColor: string;
  secondaryColor: string;
  headerBackground: string;
  tableAlternateRow: string;
  margins: { top: number; right: number; bottom: number; left: number };
  pageSize: 'A4' | 'Letter' | 'Legal';
  orientation: 'portrait' | 'landscape';
}

// Generated report
interface GeneratedReport {
  id: string;
  templateId: string;
  title: string;
  type: ReportType;
  status: ReportStatus;
  priority: ReportPriority;
  generatedAt: Date;
  generatedBy: string;
  approvedBy?: string;
  approvedAt?: Date;
  publishedAt?: Date;
  timeRange: { start: Date; end: Date };
  format: ExportFormat;
  fileSize?: number;
  filePath?: string;
  data: ReportData;
  metadata: ReportMetadata;
}

// Report data
interface ReportData {
  summary: ReportSummary;
  sections: GeneratedSection[];
  appendices?: Appendix[];
}

// Report summary
interface ReportSummary {
  title: string;
  preparedFor: string;
  preparedBy: string;
  reportingPeriod: string;
  keyHighlights: string[];
  criticalActions: string[];
  overallStatus: 'critical' | 'serious' | 'moderate' | 'stable' | 'improving';
}

// Generated section content
interface GeneratedSection {
  id: string;
  title: string;
  content: string | TableData | ChartData | MetricsData;
  pageBreakBefore?: boolean;
}

// Table data
interface TableData {
  type: 'table';
  headers: string[];
  rows: (string | number)[][];
  totals?: (string | number)[];
}

// Chart data
interface ChartData {
  type: 'chart';
  chartType: 'bar' | 'line' | 'pie' | 'area';
  labels: string[];
  datasets: { label: string; data: number[]; color?: string }[];
}

// Metrics data
interface MetricsData {
  type: 'metrics';
  items: { label: string; value: string | number; change?: number; status?: 'good' | 'warning' | 'critical' }[];
}

// Appendix
interface Appendix {
  title: string;
  content: string;
  order: number;
}

// Report metadata
interface ReportMetadata {
  disasterType?: string;
  affectedRegion?: string;
  incidentId?: string;
  version: string;
  confidentiality: 'public' | 'internal' | 'confidential' | 'restricted';
  distribution: string[];
  tags: string[];
}

// Schedule configuration
interface ScheduleConfig {
  id: string;
  templateId: string;
  name: string;
  enabled: boolean;
  frequency: 'daily' | 'weekly' | 'monthly' | 'on_event';
  time?: string;
  dayOfWeek?: number;
  dayOfMonth?: number;
  eventTrigger?: string;
  recipients: Recipient[];
  format: ExportFormat;
  lastRun?: Date;
  nextRun?: Date;
}

// Recipient
interface Recipient {
  name: string;
  email: string;
  role?: string;
  notifyMethod: 'email' | 'sms' | 'both';
}

// Default templates
const DEFAULT_TEMPLATES: ReportTemplate[] = [
  {
    id: 'situation-report',
    name: 'Situation Report (SITREP)',
    type: 'situation_report',
    description: 'Standard situation report for disaster response',
    sections: [
      { id: 's1', title: 'Executive Summary', type: 'text', order: 1, required: true },
      { id: 's2', title: 'Current Situation', type: 'text', order: 2, required: true },
      { id: 's3', title: 'Key Statistics', type: 'metrics', order: 3, required: true },
      { id: 's4', title: 'Affected Areas', type: 'table', order: 4, required: true },
      { id: 's5', title: 'Response Activities', type: 'list', order: 5, required: true },
      { id: 's6', title: 'Resource Status', type: 'table', order: 6, required: false },
      { id: 's7', title: 'Challenges & Gaps', type: 'list', order: 7, required: false },
      { id: 's8', title: 'Planned Actions', type: 'list', order: 8, required: true },
      { id: 's9', title: 'Contact Information', type: 'table', order: 9, required: true },
    ],
    defaultFormat: 'pdf',
    headerText: 'DISASTER RESPONSE SITUATION REPORT',
    footerText: 'This report is prepared for official use only.',
    styling: {
      fontFamily: 'Arial',
      fontSize: 11,
      primaryColor: '#1976D2',
      secondaryColor: '#424242',
      headerBackground: '#1976D2',
      tableAlternateRow: '#F5F5F5',
      margins: { top: 20, right: 15, bottom: 20, left: 15 },
      pageSize: 'A4',
      orientation: 'portrait',
    },
  },
  {
    id: 'damage-assessment',
    name: 'Damage Assessment Report',
    type: 'damage_assessment',
    description: 'Comprehensive damage assessment documentation',
    sections: [
      { id: 'd1', title: 'Assessment Overview', type: 'text', order: 1, required: true },
      { id: 'd2', title: 'Impact Summary', type: 'metrics', order: 2, required: true },
      { id: 'd3', title: 'Infrastructure Damage', type: 'table', order: 3, required: true },
      { id: 'd4', title: 'Housing Damage', type: 'table', order: 4, required: true },
      { id: 'd5', title: 'Agricultural Loss', type: 'table', order: 5, required: false },
      { id: 'd6', title: 'Economic Impact', type: 'chart', order: 6, required: true },
      { id: 'd7', title: 'Damage by Region', type: 'chart', order: 7, required: true },
      { id: 'd8', title: 'Photographic Evidence', type: 'image', order: 8, required: false },
      { id: 'd9', title: 'Recovery Recommendations', type: 'list', order: 9, required: true },
    ],
    defaultFormat: 'pdf',
    headerText: 'DAMAGE ASSESSMENT REPORT',
    styling: {
      fontFamily: 'Calibri',
      fontSize: 11,
      primaryColor: '#D32F2F',
      secondaryColor: '#616161',
      headerBackground: '#D32F2F',
      tableAlternateRow: '#FFEBEE',
      margins: { top: 20, right: 15, bottom: 20, left: 15 },
      pageSize: 'A4',
      orientation: 'portrait',
    },
  },
  {
    id: 'evacuation-summary',
    name: 'Evacuation Summary Report',
    type: 'evacuation_summary',
    description: 'Summary of evacuation operations',
    sections: [
      { id: 'e1', title: 'Evacuation Overview', type: 'text', order: 1, required: true },
      { id: 'e2', title: 'Evacuation Statistics', type: 'metrics', order: 2, required: true },
      { id: 'e3', title: 'Evacuation Timeline', type: 'chart', order: 3, required: true },
      { id: 'e4', title: 'Evacuation Routes Used', type: 'table', order: 4, required: true },
      { id: 'e5', title: 'Shelter Allocation', type: 'table', order: 5, required: true },
      { id: 'e6', title: 'Transport Resources', type: 'table', order: 6, required: false },
      { id: 'e7', title: 'Special Needs Population', type: 'table', order: 7, required: true },
      { id: 'e8', title: 'Challenges Faced', type: 'list', order: 8, required: false },
    ],
    defaultFormat: 'pdf',
    headerText: 'EVACUATION OPERATIONS SUMMARY',
    styling: {
      fontFamily: 'Arial',
      fontSize: 11,
      primaryColor: '#FF9800',
      secondaryColor: '#616161',
      headerBackground: '#FF9800',
      tableAlternateRow: '#FFF3E0',
      margins: { top: 20, right: 15, bottom: 20, left: 15 },
      pageSize: 'A4',
      orientation: 'portrait',
    },
  },
  {
    id: 'shelter-status',
    name: 'Shelter Status Report',
    type: 'shelter_status',
    description: 'Status of relief shelters and camps',
    sections: [
      { id: 'sh1', title: 'Shelter Overview', type: 'text', order: 1, required: true },
      { id: 'sh2', title: 'Occupancy Statistics', type: 'metrics', order: 2, required: true },
      { id: 'sh3', title: 'Shelter Directory', type: 'table', order: 3, required: true },
      { id: 'sh4', title: 'Capacity Utilization', type: 'chart', order: 4, required: true },
      { id: 'sh5', title: 'Facilities Available', type: 'table', order: 5, required: true },
      { id: 'sh6', title: 'Supply Status', type: 'table', order: 6, required: true },
      { id: 'sh7', title: 'Health & Safety', type: 'list', order: 7, required: true },
      { id: 'sh8', title: 'Staffing Levels', type: 'table', order: 8, required: false },
    ],
    defaultFormat: 'pdf',
    styling: {
      fontFamily: 'Arial',
      fontSize: 11,
      primaryColor: '#4CAF50',
      secondaryColor: '#616161',
      headerBackground: '#4CAF50',
      tableAlternateRow: '#E8F5E9',
      margins: { top: 20, right: 15, bottom: 20, left: 15 },
      pageSize: 'A4',
      orientation: 'portrait',
    },
  },
  {
    id: 'daily-briefing',
    name: 'Daily Briefing Report',
    type: 'daily_briefing',
    description: 'Daily summary for stakeholders',
    sections: [
      { id: 'db1', title: 'Daily Highlights', type: 'text', order: 1, required: true },
      { id: 'db2', title: 'Key Metrics', type: 'metrics', order: 2, required: true },
      { id: 'db3', title: 'Incidents Today', type: 'table', order: 3, required: true },
      { id: 'db4', title: 'Response Activities', type: 'list', order: 4, required: true },
      { id: 'db5', title: 'Resource Utilization', type: 'chart', order: 5, required: false },
      { id: 'db6', title: 'Tomorrow\'s Priorities', type: 'list', order: 6, required: true },
      { id: 'db7', title: 'Weather Forecast', type: 'text', order: 7, required: false },
    ],
    defaultFormat: 'pdf',
    styling: {
      fontFamily: 'Calibri',
      fontSize: 11,
      primaryColor: '#673AB7',
      secondaryColor: '#757575',
      headerBackground: '#673AB7',
      tableAlternateRow: '#EDE7F6',
      margins: { top: 20, right: 15, bottom: 20, left: 15 },
      pageSize: 'A4',
      orientation: 'portrait',
    },
  },
];

class ReportGenerationService {
  private static instance: ReportGenerationService;
  private templates: Map<string, ReportTemplate> = new Map();
  private reports: Map<string, GeneratedReport> = new Map();
  private schedules: Map<string, ScheduleConfig> = new Map();
  private listeners: ((report: GeneratedReport) => void)[] = [];

  private constructor() {
    this.initializeTemplates();
  }

  public static getInstance(): ReportGenerationService {
    if (!ReportGenerationService.instance) {
      ReportGenerationService.instance = new ReportGenerationService();
    }
    return ReportGenerationService.instance;
  }

  /**
   * Initialize default templates
   */
  private initializeTemplates(): void {
    DEFAULT_TEMPLATES.forEach((template) => {
      this.templates.set(template.id, template);
    });
  }

  /**
   * Generate report from template
   */
  public async generateReport(
    templateId: string,
    options: {
      title?: string;
      timeRange: { start: Date; end: Date };
      format?: ExportFormat;
      priority?: ReportPriority;
      metadata?: Partial<ReportMetadata>;
      customData?: Record<string, unknown>;
    }
  ): Promise<GeneratedReport> {
    const template = this.templates.get(templateId);
    if (!template) {
      throw new Error(`Template not found: ${templateId}`);
    }

    const reportId = `report-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`;
    const now = new Date();

    // Generate section content
    const sections = await this.generateSections(template, options.timeRange, options.customData);
    const summary = this.generateSummary(template, sections, options.timeRange);

    const report: GeneratedReport = {
      id: reportId,
      templateId,
      title: options.title || `${template.name} - ${this.formatDateRange(options.timeRange)}`,
      type: template.type,
      status: 'draft',
      priority: options.priority || 'normal',
      generatedAt: now,
      generatedBy: 'System',
      timeRange: options.timeRange,
      format: options.format || template.defaultFormat,
      data: {
        summary,
        sections,
      },
      metadata: {
        version: '1.0',
        confidentiality: 'internal',
        distribution: [],
        tags: [],
        ...options.metadata,
      },
    };

    this.reports.set(reportId, report);
    this.notifyListeners(report);

    return report;
  }

  /**
   * Generate sections content
   */
  private async generateSections(
    template: ReportTemplate,
    timeRange: { start: Date; end: Date },
    customData?: Record<string, unknown>
  ): Promise<GeneratedSection[]> {
    const sections: GeneratedSection[] = [];

    for (const section of template.sections.sort((a, b) => a.order - b.order)) {
      const content = await this.generateSectionContent(section, timeRange, customData);
      sections.push({
        id: section.id,
        title: section.title,
        content,
        pageBreakBefore: section.order > 1 && section.type === 'chart',
      });
    }

    return sections;
  }

  /**
   * Generate content for a section
   */
  private async generateSectionContent(
    section: ReportSection,
    timeRange: { start: Date; end: Date },
    customData?: Record<string, unknown>
  ): Promise<string | TableData | ChartData | MetricsData> {
    switch (section.type) {
      case 'text':
        return this.generateTextContent(section.id, timeRange);
      case 'table':
        return this.generateTableContent(section.id);
      case 'chart':
        return this.generateChartContent(section.id);
      case 'metrics':
        return this.generateMetricsContent(section.id);
      case 'list':
        return this.generateListContent(section.id);
      default:
        return `Content for ${section.title}`;
    }
  }

  /**
   * Generate text content
   */
  private generateTextContent(sectionId: string, timeRange: { start: Date; end: Date }): string {
    const textContent: Record<string, string> = {
      s1: `This situation report provides an overview of the disaster response operations for the period ${this.formatDateRange(timeRange)}. The response teams have been actively engaged in rescue, relief, and rehabilitation activities across affected areas.`,
      s2: `The current situation remains challenging with ongoing rescue operations in several districts. Emergency response teams are working around the clock to provide assistance to affected communities. Weather conditions are being closely monitored for any changes that may impact operations.`,
      d1: `This damage assessment report documents the extent of damage caused by the recent disaster. The assessment was conducted by teams deployed across all affected areas using standardized assessment protocols.`,
      e1: `The evacuation operations covered multiple districts and were completed within the planned timeline. Coordination between various agencies ensured smooth execution of the evacuation plan.`,
      sh1: `This report provides a comprehensive overview of all active relief shelters and their current operational status. All shelters are following standard operating procedures and health protocols.`,
      db1: `Today's operations focused on continuing relief distribution and completing pending evacuations in affected areas. Several key milestones were achieved including completion of medical camp setup.`,
    };
    return textContent[sectionId] || `Detailed information for this section covering the reporting period.`;
  }

  /**
   * Generate table content
   */
  private generateTableContent(sectionId: string): TableData {
    const tableData: Record<string, TableData> = {
      s4: {
        type: 'table',
        headers: ['District', 'Taluks Affected', 'Villages', 'Population', 'Severity'],
        rows: [
          ['Ernakulam', '8', '45', '125,000', 'High'],
          ['Thrissur', '6', '32', '85,000', 'Medium'],
          ['Wayanad', '4', '28', '65,000', 'Critical'],
          ['Kozhikode', '5', '35', '95,000', 'High'],
          ['Malappuram', '7', '40', '110,000', 'Medium'],
        ],
        totals: ['Total', '30', '180', '480,000', '-'],
      },
      s6: {
        type: 'table',
        headers: ['Resource', 'Required', 'Available', 'Deployed', 'Gap'],
        rows: [
          ['Rescue Boats', '100', '85', '80', '15'],
          ['Ambulances', '50', '45', '42', '5'],
          ['Medical Teams', '80', '75', '70', '5'],
          ['Relief Vehicles', '200', '180', '175', '20'],
          ['Water Tankers', '60', '55', '50', '5'],
        ],
      },
      s9: {
        type: 'table',
        headers: ['Role', 'Name', 'Contact', 'Email'],
        rows: [
          ['EOC Chief', 'District Collector', '+91-xxxx-xxxxxx', 'collector@nic.in'],
          ['Rescue Lead', 'SP Operations', '+91-xxxx-xxxxxx', 'sp@police.gov.in'],
          ['Medical Lead', 'CMO', '+91-xxxx-xxxxxx', 'cmo@health.gov.in'],
          ['Relief Coordinator', 'Sub Collector', '+91-xxxx-xxxxxx', 'sc@nic.in'],
        ],
      },
      d3: {
        type: 'table',
        headers: ['Infrastructure', 'Total', 'Damaged', 'Destroyed', 'Est. Cost (Cr)'],
        rows: [
          ['Roads (km)', '500', '150', '25', '45.5'],
          ['Bridges', '45', '12', '3', '28.0'],
          ['Power Lines (km)', '800', '200', '50', '15.5'],
          ['Water Pipelines (km)', '300', '80', '20', '8.0'],
          ['Schools', '120', '35', '8', '22.0'],
          ['Hospitals', '25', '5', '1', '35.0'],
        ],
      },
      d4: {
        type: 'table',
        headers: ['Category', 'Total Houses', 'Fully Damaged', 'Partially Damaged', 'Est. Cost (Cr)'],
        rows: [
          ['Pucca Houses', '25,000', '1,200', '5,500', '180.0'],
          ['Semi-Pucca', '35,000', '2,500', '8,000', '85.0'],
          ['Kutcha Houses', '15,000', '3,000', '6,000', '25.0'],
          ['Apartments', '8,000', '500', '2,000', '120.0'],
        ],
      },
      e4: {
        type: 'table',
        headers: ['Route ID', 'Origin', 'Destination', 'Distance (km)', 'Evacuees'],
        rows: [
          ['R-001', 'Aluva', 'Govt School Shelter', '12', '2,500'],
          ['R-002', 'Paravur', 'Community Hall', '8', '1,800'],
          ['R-003', 'Kalady', 'College Campus', '15', '3,200'],
          ['R-004', 'Perumbavoor', 'Stadium Complex', '10', '2,100'],
          ['R-005', 'Angamaly', 'Marriage Hall', '6', '1,500'],
        ],
      },
      e5: {
        type: 'table',
        headers: ['Shelter Name', 'Location', 'Capacity', 'Occupancy', 'Status'],
        rows: [
          ['Govt Higher Secondary', 'Aluva', '500', '485', 'Near Full'],
          ['Town Hall', 'Ernakulam', '800', '650', 'Open'],
          ['Sports Complex', 'Thrissur', '1,200', '1,180', 'Full'],
          ['Community Center', 'Thodupuzha', '400', '320', 'Open'],
          ['Relief Camp 1', 'Munnar', '600', '550', 'Open'],
        ],
      },
      sh3: {
        type: 'table',
        headers: ['Shelter ID', 'Name', 'District', 'Capacity', 'Current', 'Available'],
        rows: [
          ['SH-001', 'Govt School Camp', 'Ernakulam', '500', '450', '50'],
          ['SH-002', 'Community Center', 'Thrissur', '300', '280', '20'],
          ['SH-003', 'Stadium Complex', 'Wayanad', '800', '750', '50'],
          ['SH-004', 'College Hostel', 'Kozhikode', '400', '380', '20'],
          ['SH-005', 'Marriage Hall', 'Malappuram', '350', '200', '150'],
        ],
      },
      sh5: {
        type: 'table',
        headers: ['Facility', 'SH-001', 'SH-002', 'SH-003', 'SH-004', 'SH-005'],
        rows: [
          ['Drinking Water', '✓', '✓', '✓', '✓', '✓'],
          ['Toilets', '✓', '✓', '✓', '✓', '✓'],
          ['Medical Room', '✓', '✓', '✓', '-', '✓'],
          ['Kitchen', '✓', '✓', '✓', '✓', '✓'],
          ['Generator', '✓', '-', '✓', '✓', '-'],
          ['CCTV', '✓', '-', '✓', '-', '-'],
        ],
      },
      sh6: {
        type: 'table',
        headers: ['Item', 'Required', 'Available', 'Stock Days', 'Status'],
        rows: [
          ['Rice (kg)', '5,000', '4,500', '6', 'Adequate'],
          ['Drinking Water (L)', '10,000', '8,000', '3', 'Low'],
          ['Medicines', '500 kits', '450', '5', 'Adequate'],
          ['Blankets', '2,000', '1,800', '-', 'Adequate'],
          ['Sanitary Kits', '1,500', '800', '4', 'Critical'],
        ],
      },
      sh8: {
        type: 'table',
        headers: ['Role', 'Required', 'Present', 'Gap', 'Shift Coverage'],
        rows: [
          ['Camp Manager', '5', '5', '0', '24x7'],
          ['Medical Staff', '15', '12', '3', '24x7'],
          ['Kitchen Staff', '25', '22', '3', '18 hrs'],
          ['Security', '20', '18', '2', '24x7'],
          ['Volunteers', '50', '45', '5', '12 hrs'],
        ],
      },
      db3: {
        type: 'table',
        headers: ['Time', 'Incident Type', 'Location', 'Status', 'Response'],
        rows: [
          ['06:30', 'Flood Alert', 'Aluva', 'Resolved', 'Evacuation Complete'],
          ['08:15', 'Medical Emergency', 'Shelter 3', 'Resolved', 'Patient Transferred'],
          ['10:45', 'Supply Shortage', 'Camp 2', 'In Progress', 'Supplies Dispatched'],
          ['14:20', 'Road Blockage', 'NH-66', 'Resolved', 'Cleared by PWD'],
          ['16:00', 'Power Outage', 'Shelter 1', 'Resolved', 'Generator Activated'],
        ],
      },
    };
    return tableData[sectionId] || {
      type: 'table',
      headers: ['Column 1', 'Column 2', 'Column 3'],
      rows: [
        ['Data 1', 'Data 2', 'Data 3'],
        ['Data 4', 'Data 5', 'Data 6'],
      ],
    };
  }

  /**
   * Generate chart content
   */
  private generateChartContent(sectionId: string): ChartData {
    const chartData: Record<string, ChartData> = {
      d6: {
        type: 'chart',
        chartType: 'pie',
        labels: ['Infrastructure', 'Housing', 'Agriculture', 'Industry', 'Others'],
        datasets: [{
          label: 'Economic Impact (Cr)',
          data: [150, 400, 200, 80, 50],
          color: '#1976D2',
        }],
      },
      d7: {
        type: 'chart',
        chartType: 'bar',
        labels: ['Ernakulam', 'Thrissur', 'Wayanad', 'Kozhikode', 'Malappuram'],
        datasets: [{
          label: 'Damage Estimate (Cr)',
          data: [250, 180, 320, 150, 120],
          color: '#D32F2F',
        }],
      },
      e3: {
        type: 'chart',
        chartType: 'line',
        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
        datasets: [{
          label: 'People Evacuated',
          data: [5000, 15000, 28000, 42000, 55000, 62000, 68000],
          color: '#FF9800',
        }],
      },
      sh4: {
        type: 'chart',
        chartType: 'bar',
        labels: ['SH-001', 'SH-002', 'SH-003', 'SH-004', 'SH-005'],
        datasets: [
          { label: 'Capacity', data: [500, 300, 800, 400, 350], color: '#1976D2' },
          { label: 'Occupied', data: [450, 280, 750, 380, 200], color: '#4CAF50' },
        ],
      },
      db5: {
        type: 'chart',
        chartType: 'pie',
        labels: ['Rescue', 'Medical', 'Relief', 'Logistics', 'Admin'],
        datasets: [{
          label: 'Resource Distribution',
          data: [30, 25, 25, 15, 5],
        }],
      },
    };
    return chartData[sectionId] || {
      type: 'chart',
      chartType: 'bar',
      labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
      datasets: [{ label: 'Value', data: [10, 20, 30, 40, 50] }],
    };
  }

  /**
   * Generate metrics content
   */
  private generateMetricsContent(sectionId: string): MetricsData {
    const metricsData: Record<string, MetricsData> = {
      s3: {
        type: 'metrics',
        items: [
          { label: 'Districts Affected', value: 8, status: 'critical' },
          { label: 'Population Impacted', value: '4.8L', change: 15, status: 'warning' },
          { label: 'People Evacuated', value: '68,000', change: 8, status: 'good' },
          { label: 'Relief Camps Active', value: 125, status: 'good' },
          { label: 'Rescue Operations', value: 45, change: -5, status: 'good' },
          { label: 'Medical Camps', value: 32, status: 'good' },
        ],
      },
      d2: {
        type: 'metrics',
        items: [
          { label: 'Total Damage (Est.)', value: '₹880 Cr', status: 'critical' },
          { label: 'Houses Damaged', value: '28,200', status: 'critical' },
          { label: 'Roads Damaged (km)', value: 175, status: 'warning' },
          { label: 'Crop Loss (Ha)', value: '12,500', status: 'warning' },
          { label: 'Livestock Lost', value: '8,200', status: 'warning' },
          { label: 'Businesses Affected', value: '3,500', status: 'warning' },
        ],
      },
      e2: {
        type: 'metrics',
        items: [
          { label: 'Total Evacuated', value: '68,000', status: 'good' },
          { label: 'Evacuation Routes', value: 25, status: 'good' },
          { label: 'Shelters Used', value: 48, status: 'good' },
          { label: 'Vehicles Deployed', value: 350, status: 'good' },
          { label: 'Special Needs', value: '2,500', status: 'warning' },
          { label: 'Avg Time (hrs)', value: '4.5', status: 'good' },
        ],
      },
      sh2: {
        type: 'metrics',
        items: [
          { label: 'Active Shelters', value: 48, status: 'good' },
          { label: 'Total Capacity', value: '25,000', status: 'good' },
          { label: 'Current Occupancy', value: '21,500', change: 3, status: 'warning' },
          { label: 'Occupancy Rate', value: '86%', status: 'warning' },
          { label: 'Medical Staff', value: 180, status: 'good' },
          { label: 'Volunteers', value: 850, status: 'good' },
        ],
      },
      db2: {
        type: 'metrics',
        items: [
          { label: 'Active Alerts', value: 12, change: -3, status: 'warning' },
          { label: 'Rescue Completed', value: 45, change: 8, status: 'good' },
          { label: 'Relief Distributed', value: '5,200 kits', change: 12, status: 'good' },
          { label: 'Shelters Active', value: 48, status: 'good' },
          { label: 'Volunteers Active', value: 850, change: 50, status: 'good' },
          { label: 'Roads Cleared', value: 15, change: 5, status: 'good' },
        ],
      },
    };
    return metricsData[sectionId] || {
      type: 'metrics',
      items: [
        { label: 'Metric 1', value: 100, status: 'good' },
        { label: 'Metric 2', value: 200, status: 'warning' },
      ],
    };
  }

  /**
   * Generate list content
   */
  private generateListContent(sectionId: string): string {
    const listContent: Record<string, string[]> = {
      s5: [
        'Deployed 45 NDRF teams for search and rescue operations',
        'Established 32 medical camps across affected districts',
        'Distributed relief kits to 25,000 families',
        'Restored power supply in 60% of affected areas',
        'Cleared 15 major road blockages for relief movement',
        'Airlifted 500 stranded people from isolated areas',
      ],
      s7: [
        'Shortage of rescue boats in remote areas',
        'Limited helicopter availability for airlifting',
        'Communication blackout in 3 taluks',
        'Fuel shortage affecting relief vehicle operations',
        'Need for additional medical specialists in camps',
      ],
      s8: [
        'Deploy additional rescue teams to Wayanad district',
        'Expedite restoration of communication networks',
        'Scale up relief material procurement',
        'Establish additional medical camps in remote areas',
        'Coordinate with neighboring states for resource support',
      ],
      d9: [
        'Prioritize restoration of critical infrastructure',
        'Implement housing reconstruction scheme',
        'Provide agricultural relief packages',
        'Support affected businesses with soft loans',
        'Strengthen early warning systems',
        'Update disaster management plans based on lessons learned',
      ],
      e8: [
        'Traffic congestion on primary evacuation routes',
        'Some families reluctant to leave homes',
        'Limited accessibility to remote hamlets',
        'Coordination challenges during night operations',
      ],
      sh7: [
        'Daily health checkups for all camp residents',
        'Regular sanitization of common areas',
        'Separate areas for suspected COVID cases',
        'Clean drinking water supply maintained',
        'Waste management systems operational',
        'Fire safety equipment checked daily',
      ],
      db4: [
        'Completed evacuation of remaining 500 families',
        'Distributed 5,200 relief kits',
        'Restored water supply in 12 localities',
        'Conducted medical screening for 3,000 people',
        'Cleared debris from 5 major roads',
      ],
      db6: [
        'Focus on restoration of remaining infrastructure',
        'Scale up relief distribution in interior areas',
        'Begin damage assessment documentation',
        'Coordinate with state for additional resources',
        'Review and update response strategies',
      ],
    };
    const items = listContent[sectionId] || ['Item 1', 'Item 2', 'Item 3'];
    return items.map((item, i) => `${i + 1}. ${item}`).join('\n');
  }

  /**
   * Generate report summary
   */
  private generateSummary(
    template: ReportTemplate,
    sections: GeneratedSection[],
    timeRange: { start: Date; end: Date }
  ): ReportSummary {
    return {
      title: template.name,
      preparedFor: 'State Disaster Management Authority',
      preparedBy: 'Emergency Operations Center',
      reportingPeriod: this.formatDateRange(timeRange),
      keyHighlights: [
        'Rescue operations progressing well with 95% completion',
        'All relief shelters operational with adequate supplies',
        'Medical camps providing continuous healthcare',
        'Infrastructure restoration underway',
      ],
      criticalActions: [
        'Deploy additional resources to critically affected areas',
        'Expedite relief material procurement',
        'Enhance coordination with neighboring districts',
      ],
      overallStatus: 'serious',
    };
  }

  /**
   * Export report
   */
  public async exportReport(reportId: string, format?: ExportFormat): Promise<Blob> {
    const report = this.reports.get(reportId);
    if (!report) {
      throw new Error(`Report not found: ${reportId}`);
    }

    const exportFormat = format || report.format;
    const content = this.formatReportContent(report, exportFormat);
    
    const mimeTypes: Record<ExportFormat, string> = {
      pdf: 'application/pdf',
      excel: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      csv: 'text/csv',
      word: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      html: 'text/html',
      json: 'application/json',
    };

    return new Blob([content], { type: mimeTypes[exportFormat] });
  }

  /**
   * Format report content
   */
  private formatReportContent(report: GeneratedReport, format: ExportFormat): string {
    switch (format) {
      case 'json':
        return JSON.stringify(report, null, 2);
      case 'html':
        return this.generateHtmlReport(report);
      case 'csv':
        return this.generateCsvReport(report);
      default:
        return JSON.stringify(report.data, null, 2);
    }
  }

  /**
   * Generate HTML report
   */
  private generateHtmlReport(report: GeneratedReport): string {
    const template = this.templates.get(report.templateId);
    const styling = template?.styling || DEFAULT_TEMPLATES[0].styling;

    let html = `
<!DOCTYPE html>
<html>
<head>
  <title>${report.title}</title>
  <style>
    body { font-family: ${styling.fontFamily}, sans-serif; font-size: ${styling.fontSize}px; margin: 40px; }
    h1 { color: ${styling.primaryColor}; border-bottom: 2px solid ${styling.primaryColor}; padding-bottom: 10px; }
    h2 { color: ${styling.secondaryColor}; margin-top: 30px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th { background: ${styling.headerBackground}; color: white; padding: 10px; text-align: left; }
    td { padding: 8px; border-bottom: 1px solid #ddd; }
    tr:nth-child(even) { background: ${styling.tableAlternateRow}; }
    .metric { display: inline-block; padding: 15px; margin: 10px; background: #f5f5f5; border-radius: 8px; }
    .metric-value { font-size: 24px; font-weight: bold; color: ${styling.primaryColor}; }
    .summary-box { background: #f0f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; }
  </style>
</head>
<body>
  <h1>${report.title}</h1>
  <div class="summary-box">
    <p><strong>Prepared for:</strong> ${report.data.summary.preparedFor}</p>
    <p><strong>Reporting Period:</strong> ${report.data.summary.reportingPeriod}</p>
    <p><strong>Generated:</strong> ${report.generatedAt.toLocaleString()}</p>
  </div>
`;

    for (const section of report.data.sections) {
      html += `<h2>${section.title}</h2>`;
      
      if (typeof section.content === 'string') {
        html += `<p>${section.content.replace(/\n/g, '<br>')}</p>`;
      } else if ((section.content as TableData).type === 'table') {
        const table = section.content as TableData;
        html += '<table><tr>';
        table.headers.forEach((h) => { html += `<th>${h}</th>`; });
        html += '</tr>';
        table.rows.forEach((row) => {
          html += '<tr>';
          row.forEach((cell) => { html += `<td>${cell}</td>`; });
          html += '</tr>';
        });
        html += '</table>';
      } else if ((section.content as MetricsData).type === 'metrics') {
        const metrics = section.content as MetricsData;
        html += '<div>';
        metrics.items.forEach((item) => {
          html += `<div class="metric"><div class="metric-value">${item.value}</div><div>${item.label}</div></div>`;
        });
        html += '</div>';
      }
    }

    html += '</body></html>';
    return html;
  }

  /**
   * Generate CSV report
   */
  private generateCsvReport(report: GeneratedReport): string {
    let csv = `Report: ${report.title}\nGenerated: ${report.generatedAt.toISOString()}\n\n`;

    for (const section of report.data.sections) {
      csv += `\n${section.title}\n`;
      
      if ((section.content as TableData).type === 'table') {
        const table = section.content as TableData;
        csv += table.headers.join(',') + '\n';
        table.rows.forEach((row) => {
          csv += row.join(',') + '\n';
        });
      }
    }

    return csv;
  }

  /**
   * Get all templates
   */
  public getTemplates(): ReportTemplate[] {
    return Array.from(this.templates.values());
  }

  /**
   * Get template by ID
   */
  public getTemplate(templateId: string): ReportTemplate | undefined {
    return this.templates.get(templateId);
  }

  /**
   * Get report by ID
   */
  public getReport(reportId: string): GeneratedReport | undefined {
    return this.reports.get(reportId);
  }

  /**
   * Get all reports
   */
  public getAllReports(): GeneratedReport[] {
    return Array.from(this.reports.values());
  }

  /**
   * Update report status
   */
  public updateReportStatus(reportId: string, status: ReportStatus, approver?: string): boolean {
    const report = this.reports.get(reportId);
    if (!report) return false;

    report.status = status;
    if (status === 'approved' && approver) {
      report.approvedBy = approver;
      report.approvedAt = new Date();
    }
    if (status === 'published') {
      report.publishedAt = new Date();
    }

    this.notifyListeners(report);
    return true;
  }

  /**
   * Schedule report
   */
  public scheduleReport(config: Omit<ScheduleConfig, 'id' | 'lastRun' | 'nextRun'>): ScheduleConfig {
    const id = `schedule-${Date.now()}`;
    const schedule: ScheduleConfig = {
      ...config,
      id,
      nextRun: this.calculateNextRun(config),
    };
    this.schedules.set(id, schedule);
    return schedule;
  }

  /**
   * Calculate next run time
   */
  private calculateNextRun(config: Omit<ScheduleConfig, 'id' | 'lastRun' | 'nextRun'>): Date {
    const now = new Date();
    const next = new Date(now);

    if (config.time) {
      const [hours, minutes] = config.time.split(':').map(Number);
      next.setHours(hours, minutes, 0, 0);
    }

    switch (config.frequency) {
      case 'daily':
        if (next <= now) next.setDate(next.getDate() + 1);
        break;
      case 'weekly':
        if (config.dayOfWeek !== undefined) {
          const daysUntil = (config.dayOfWeek - now.getDay() + 7) % 7 || 7;
          next.setDate(next.getDate() + daysUntil);
        }
        break;
      case 'monthly':
        if (config.dayOfMonth) {
          next.setDate(config.dayOfMonth);
          if (next <= now) next.setMonth(next.getMonth() + 1);
        }
        break;
    }

    return next;
  }

  /**
   * Format date range
   */
  private formatDateRange(range: { start: Date; end: Date }): string {
    const options: Intl.DateTimeFormatOptions = { day: '2-digit', month: 'short', year: 'numeric' };
    return `${range.start.toLocaleDateString('en-IN', options)} - ${range.end.toLocaleDateString('en-IN', options)}`;
  }

  /**
   * Subscribe to report updates
   */
  public subscribe(callback: (report: GeneratedReport) => void): () => void {
    this.listeners.push(callback);
    return () => {
      const index = this.listeners.indexOf(callback);
      if (index > -1) this.listeners.splice(index, 1);
    };
  }

  /**
   * Notify listeners
   */
  private notifyListeners(report: GeneratedReport): void {
    this.listeners.forEach((callback) => callback(report));
  }
}

export const reportGenerationService = ReportGenerationService.getInstance();
export type {
  ReportType,
  ExportFormat,
  ReportStatus,
  ReportPriority,
  ReportTemplate,
  ReportSection,
  SectionConfig,
  TableColumn,
  ReportStyling,
  GeneratedReport,
  ReportData,
  ReportSummary,
  GeneratedSection,
  TableData,
  ChartData,
  MetricsData,
  Appendix,
  ReportMetadata,
  ScheduleConfig,
  Recipient,
};
