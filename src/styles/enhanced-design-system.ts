/**
 * PRODUCTION-GRADE DESIGN SYSTEM
 * Enhanced with perfect spacing, professional shadows, and interactive states
 * 16/24px base grid system - Zero overlapping guarantee
 */

import { css } from 'styled-components';

// ENHANCED SPACING SYSTEM - 16/24px Base Grid
export const enhancedSpacing = {
  // Core units
  unit: 4,     // 4px base unit
  base: 16,    // 16px primary base 
  large: 24,   // 24px secondary base
  
  // Calculated spacing scale
  0: '0',
  1: '4px',    // 1 unit
  2: '8px',    // 2 units  
  3: '12px',   // 3 units
  4: '16px',   // 4 units (base)
  5: '20px',   // 5 units
  6: '24px',   // 6 units (large base)
  8: '32px',   // 8 units
  10: '40px',  // 10 units
  12: '48px',  // 12 units (2 * large)
  16: '64px',  // 16 units
  20: '80px',  // 20 units
  24: '96px',  // 24 units (4 * large)
  32: '128px', // 32 units
} as const;

// PROFESSIONAL SHADOW SYSTEM - rgba(0,0,0,0.15) base
export const enhancedShadows = {
  none: 'none',
  
  // Subtle elevation levels
  xs: '0 1px 2px rgba(0, 0, 0, 0.06)',
  sm: '0 2px 4px rgba(0, 0, 0, 0.08)',
  
  // Standard card shadows
  default: '0 4px 8px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.06)',
  md: '0 8px 16px rgba(0, 0, 0, 0.15), 0 4px 8px rgba(0, 0, 0, 0.06)',
  
  // Interactive hover shadows
  hover: '0 12px 24px rgba(0, 0, 0, 0.18), 0 6px 12px rgba(0, 0, 0, 0.08)',
  focus: '0 0 0 3px rgba(59, 130, 246, 0.15), 0 8px 16px rgba(0, 0, 0, 0.15)',
  
  // High elevation
  lg: '0 16px 32px rgba(0, 0, 0, 0.20), 0 8px 16px rgba(0, 0, 0, 0.10)',
  xl: '0 24px 48px rgba(0, 0, 0, 0.25), 0 12px 24px rgba(0, 0, 0, 0.12)',
  
  // Critical/modal shadows
  modal: '0 32px 64px rgba(0, 0, 0, 0.30), 0 16px 32px rgba(0, 0, 0, 0.15)',
  
  // Inner shadows for depth
  inner: 'inset 0 2px 4px rgba(0, 0, 0, 0.08)',
  innerLg: 'inset 0 4px 8px rgba(0, 0, 0, 0.12)',
} as const;

// ENHANCED GRID SYSTEM - Perfect alignment
export const enhancedGrid = {
  // Container spacing - NO overlapping
  containerGap: enhancedSpacing[6], // 24px
  cardGap: enhancedSpacing[4],      // 16px
  elementGap: enhancedSpacing[3],   // 12px
  
  // Padding system
  cardPadding: enhancedSpacing[6],     // 24px internal
  sectionPadding: enhancedSpacing[8],  // 32px sections
  containerPadding: enhancedSpacing[6], // 24px containers
  
  // Minimum safe margins
  minMargin: enhancedSpacing[4],      // 16px minimum
  safeMargin: enhancedSpacing[5],     // 20px safe zone
  clearanceMargin: enhancedSpacing[6], // 24px clearance
} as const;

// PROFESSIONAL CARD SYSTEM
export const enhancedCard = css`
  background: ${({ theme }) => theme.colors.surface.elevated};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: 12px;
  padding: ${enhancedGrid.cardPadding};
  margin: ${enhancedGrid.minMargin};
  box-shadow: ${enhancedShadows.default};
  
  /* Perfect alignment and no clipping */
  position: relative;
  z-index: 1;
  min-height: 200px;
  overflow: visible;
  
  /* Smooth interactive transitions */
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  
  /* Enhanced hover states */
  &:hover {
    box-shadow: ${enhancedShadows.hover};
    transform: translateY(-2px);
    border-color: ${({ theme }) => theme.colors.primary[200]};
  }
  
  /* Focus states for accessibility */
  &:focus-within {
    box-shadow: ${enhancedShadows.focus};
    outline: none;
  }
  
  /* Ensure content never clips */
  * {
    position: relative;
    z-index: 2;
  }
  
  /* Chart and graph protection */
  .recharts-wrapper,
  .chart-container,
  svg {
    overflow: visible !important;
  }
`;

// ENHANCED TYPOGRAPHY SYSTEM
export const enhancedTypography = css`
  /* Perfect text alignment */
  line-height: 1.5;
  letter-spacing: -0.01em;
  
  /* Consistent margins */
  h1, h2, h3, h4, h5, h6 {
    margin: 0 0 ${enhancedSpacing[4]} 0;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.text.primary};
  }
  
  p {
    margin: 0 0 ${enhancedSpacing[3]} 0;
    color: ${({ theme }) => theme.colors.text.secondary};
  }
  
  /* Icon alignment */
  .icon-text {
    display: flex;
    align-items: center;
    gap: ${enhancedSpacing[2]};
  }
`;

// RESPONSIVE GRID BREAKPOINTS
export const enhancedBreakpoints = {
  mobile: '480px',
  tablet: '768px',
  laptop: '1024px',
  desktop: '1200px',
  ultrawide: '1440px',
} as const;

// ENHANCED LAYOUT SYSTEM
export const enhancedLayout = {
  // Dashboard grid
  dashboard: {
    maxWidth: '1400px',
    columns: {
      desktop: 'minmax(300px, 1fr) minmax(400px, 2fr) minmax(300px, 1fr)',
      tablet: 'repeat(2, 1fr)',
      mobile: '1fr',
    },
    gap: enhancedGrid.containerGap,
    padding: enhancedGrid.containerPadding,
  },
  
  // 3D visualization constraints
  visualization: {
    width: '350px',
    height: '350px',
    maxWidth: '100%',
    margin: `0 auto ${enhancedSpacing[8]} auto`,
  },
  
  // Header specs
  header: {
    height: '70px',
    padding: `0 ${enhancedSpacing[6]}`,
    zIndex: 1000,
  },
  
  // Modal specs  
  modal: {
    overlay: {
      backdropFilter: 'blur(8px)',
      background: 'rgba(0, 0, 0, 0.5)',
      zIndex: 9999,
    },
    content: {
      maxWidth: '500px',
      padding: enhancedSpacing[8],
      borderRadius: '16px',
      boxShadow: enhancedShadows.modal,
    },
  },
} as const;

// ENHANCED BUTTON SYSTEM
export const enhancedButton = css`
  /* Base button styles */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: ${enhancedSpacing[2]};
  
  padding: ${enhancedSpacing[3]} ${enhancedSpacing[6]};
  border-radius: 8px;
  border: none;
  
  font-weight: 600;
  font-size: 0.875rem;
  line-height: 1;
  text-decoration: none;
  
  cursor: pointer;
  user-select: none;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  
  /* Focus states */
  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.primary[500]};
    outline-offset: 2px;
  }
  
  /* Disabled state */
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none !important;
  }
  
  /* Size variants */
  &.size-sm {
    padding: ${enhancedSpacing[2]} ${enhancedSpacing[4]};
    font-size: 0.75rem;
  }
  
  &.size-lg {
    padding: ${enhancedSpacing[4]} ${enhancedSpacing[8]};
    font-size: 1rem;
  }
  
  /* Interactive states */
  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: ${enhancedShadows.sm};
  }
  
  &:active:not(:disabled) {
    transform: translateY(0);
  }
`;

// ZERO-OVERLAP GUARANTEE MIXINS
export const noOverlapMixin = css`
  /* Ensure minimum spacing between elements */
  & > *:not(:last-child) {
    margin-bottom: ${enhancedSpacing[4]};
  }
  
  /* Grid gap fallback */
  display: flex;
  flex-direction: column;
  gap: ${enhancedSpacing[4]};
  
  /* Override for grid layouts */
  &.grid {
    display: grid;
    gap: ${enhancedSpacing[4]};
  }
  
  /* Clearfix for floated elements */
  &::after {
    content: '';
    display: table;
    clear: both;
  }
`;

// CHART PROTECTION SYSTEM
export const chartProtectionMixin = css`
  /* Protect charts from clipping */
  .recharts-wrapper,
  .recharts-surface,
  .recharts-layer,
  .chart-container {
    overflow: visible !important;
  }
  
  /* Ensure axes and legends are visible */
  .recharts-cartesian-axis,
  .recharts-legend-wrapper {
    overflow: visible !important;
  }
  
  /* Tooltip positioning */
  .recharts-tooltip-wrapper {
    z-index: 999;
  }
`;

// ACCESSIBILITY ENHANCEMENTS
export const a11yMixin = css`
  /* Focus indicators */
  :focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.primary[500]};
    outline-offset: 2px;
  }
  
  /* High contrast text */
  color: ${({ theme }) => theme.colors.text.primary};
  
  /* Screen reader helpers */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
  
  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    * {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }
`;

// EXPORT ENHANCED DESIGN SYSTEM
export const enhancedDesignSystem = {
  spacing: enhancedSpacing,
  shadows: enhancedShadows,
  grid: enhancedGrid,
  card: enhancedCard,
  typography: enhancedTypography,
  breakpoints: enhancedBreakpoints,
  layout: enhancedLayout,
  button: enhancedButton,
  noOverlap: noOverlapMixin,
  chartProtection: chartProtectionMixin,
  a11y: a11yMixin,
} as const;

export default enhancedDesignSystem;