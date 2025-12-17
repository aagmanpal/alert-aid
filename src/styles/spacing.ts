// Design System - 24px Base Unit Spacing
export const spacing = {
  // Base unit: 24px
  base: '24px',
  
  // Derived spacing scale
  xs: '4px',    // 0.167 * base
  sm: '8px',    // 0.333 * base  
  md: '12px',   // 0.5 * base
  lg: '16px',   // 0.667 * base
  xl: '24px',   // 1 * base (same as base)
  '2xl': '32px', // 1.333 * base
  '3xl': '48px', // 2 * base
  '4xl': '72px', // 3 * base
  '5xl': '96px', // 4 * base
} as const;

// Grid spacing
export const gridSpacing = {
  // Container gaps
  container: '24px',
  cardInterior: '16px',
  
  // Minimum margins
  minMargin: '16px',
  cardPadding: '20px',
  
  // Component spacing
  componentGap: '24px',
  elementGap: '16px',
} as const;

// Responsive breakpoints
export const breakpoints = {
  mobile: '767px',
  tablet: '768px',
  tabletMax: '1199px', 
  desktop: '1200px',
} as const;

// Layout utilities
export const layout = {
  // 3D visualization constraints
  visualization: {
    width: '350px',
    height: '350px',
  },
  
  // Card standardization
  card: {
    minHeight: '200px',
    padding: spacing.xl, // 24px
    margin: spacing.lg,  // 16px minimum
    borderRadius: '12px',
  },
  
  // Typography spacing
  typography: {
    lineHeight: '1.5',
    marginBottom: spacing.lg, // 16px
  },
} as const;