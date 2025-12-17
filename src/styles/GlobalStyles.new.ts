import { createGlobalStyle } from 'styled-components';
import { Theme } from './theme';
import { spacing, gridSpacing, breakpoints, layout } from './spacing';

export const GlobalStyles = createGlobalStyle<{ theme: Theme }>`
  /* Import Inter font from Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html {
    font-size: 16px;
    scroll-behavior: smooth;
    background: ${({ theme }) => theme.colors.background.primary};
  }

  body {
    font-family: ${({ theme }) => theme.typography.fontFamily.primary};
    background: ${({ theme }) => theme.colors.background.primary};
    color: ${({ theme }) => theme.colors.text.primary};
    line-height: ${layout.typography.lineHeight};
    font-weight: ${({ theme }) => theme.typography.fontWeight.normal};
    font-size: ${({ theme }) => theme.typography.fontSize.body};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    overflow-x: hidden;
  }

  #root {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: ${({ theme }) => theme.colors.background.primary};
  }

  /* 24px Base Spacing System */
  .container {
    display: grid;
    gap: ${gridSpacing.container};
    padding: ${spacing.xl};
    max-width: 100%;
    margin: 0 auto;
  }

  .card {
    padding: ${layout.card.padding};
    margin: ${layout.card.margin};
    min-height: ${layout.card.minHeight};
    border-radius: ${layout.card.borderRadius};
    background: ${({ theme }) => theme.colors.surface.elevated};
    border: 1px solid ${({ theme }) => theme.colors.border.default};
    transition: all 0.3s ease;
    
    &:hover {
      box-shadow: ${({ theme }) => theme.shadows.medium};
    }
  }

  /* Responsive Dashboard Grid - NO OVERLAPPING */
  .dashboard-grid {
    display: grid;
    gap: ${gridSpacing.container};
    padding: ${spacing.xl};
    width: 100%;
    
    /* Desktop: 3-column layout */
    @media (min-width: ${breakpoints.desktop}) {
      grid-template-columns: repeat(3, 1fr);
      max-width: 1400px;
      margin: 0 auto;
    }
    
    /* Tablet: 2-column layout */
    @media (min-width: ${breakpoints.tablet}) and (max-width: ${breakpoints.tabletMax}) {
      grid-template-columns: repeat(2, 1fr);
      padding: ${spacing.lg};
    }
    
    /* Mobile: Single column */
    @media (max-width: ${breakpoints.mobile}) {
      grid-template-columns: 1fr;
      padding: ${spacing.md};
      gap: ${spacing.lg};
    }
  }

  /* Ensure NO OVERLAPPING - Minimum spacing enforcement */
  .dashboard-card {
    margin: ${gridSpacing.minMargin};
    padding: ${gridSpacing.cardPadding};
    position: relative;
    z-index: 1;
    
    /* Ensure minimum space between cards */
    &:not(:last-child) {
      margin-bottom: ${gridSpacing.minMargin};
    }
  }

  /* 3D Visualization Constraints - Reduced size */
  .visualization-container {
    width: ${layout.visualization.width};
    height: ${layout.visualization.height};
    margin: 0 auto ${spacing.xl} auto;
    max-width: 100%;
    
    @media (max-width: ${breakpoints.mobile}) {
      width: 100%;
      max-width: 280px;
      height: 280px;
    }
  }

  /* Typography with consistent 16px spacing */
  h1, h2, h3, h4, h5, h6 {
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    color: ${({ theme }) => theme.colors.text.primary};
    margin-bottom: ${layout.typography.marginBottom};
    line-height: ${layout.typography.lineHeight};
  }

  h1 { font-size: ${({ theme }) => theme.typography.fontSize['4xl']}; }
  h2 { font-size: ${({ theme }) => theme.typography.fontSize['3xl']}; }
  h3 { font-size: ${({ theme }) => theme.typography.fontSize['2xl']}; }
  h4 { font-size: ${({ theme }) => theme.typography.fontSize.xl}; }
  h5 { font-size: ${({ theme }) => theme.typography.fontSize.large}; }
  h6 { font-size: ${({ theme }) => theme.typography.fontSize.body}; }

  p {
    margin-bottom: ${layout.typography.marginBottom};
    color: ${({ theme }) => theme.colors.text.secondary};
    line-height: ${layout.typography.lineHeight};
  }

  /* Interactive elements with smooth transitions */
  a {
    color: ${({ theme }) => theme.colors.primary[600]};
    text-decoration: none;
    transition: all 0.3s ease;
    
    &:hover {
      color: ${({ theme }) => theme.colors.primary[500]};
    }
  }

  button {
    font-family: inherit;
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 44px; /* Touch target minimum */
    padding: ${spacing.sm} ${spacing.lg};
    border: none;
    border-radius: 8px;
    
    &:hover {
      transform: translateY(-1px);
    }
    
    &:active {
      transform: translateY(0);
    }
  }

  input, textarea, select {
    font-family: inherit;
    min-height: 44px;
    padding: ${spacing.sm} ${spacing.md};
    border-radius: 6px;
    border: 1px solid ${({ theme }) => theme.colors.border.default};
    background: ${({ theme }) => theme.colors.surface.default};
    color: ${({ theme }) => theme.colors.text.primary};
    transition: all 0.3s ease;
    
    &:focus {
      border-color: ${({ theme }) => theme.colors.primary[600]};
      box-shadow: 0 0 0 3px ${({ theme }) => theme.colors.primary[600]}20;
    }
  }

  /* Smooth transitions for all interactive elements */
  * {
    transition: all 0.3s ease;
  }

  /* Dark mode scrollbar */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${({ theme }) => theme.colors.surface.default};
  }

  ::-webkit-scrollbar-thumb {
    background: ${({ theme }) => theme.colors.border.medium};
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: ${({ theme }) => theme.colors.border.strong};
  }

  /* Selection styles */
  ::selection {
    background: ${({ theme }) => theme.colors.primary[600]}40;
    color: ${({ theme }) => theme.colors.text.primary};
  }

  /* WCAG 2.1 AA Compliance - Focus styles */
  *:focus {
    outline: 2px solid ${({ theme }) => theme.colors.primary[600]};
    outline-offset: 2px;
  }

  *:focus:not(.focus-visible) {
    outline: none;
  }

  /* Keyboard navigation */
  .keyboard-focused *:focus {
    outline: 2px solid ${({ theme }) => theme.colors.primary[600]} !important;
    outline-offset: 2px !important;
  }

  /* High contrast mode support */
  @media (prefers-contrast: high) {
    * {
      border-color: ${({ theme }) => theme.colors.text.primary} !important;
    }
  }

  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    * {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }

  /* Loading states - Skeleton screens */
  .skeleton {
    background: linear-gradient(
      90deg,
      ${({ theme }) => theme.colors.surface.default} 25%,
      ${({ theme }) => theme.colors.surface.hover} 50%,
      ${({ theme }) => theme.colors.surface.default} 75%
    );
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
  }

  @keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  /* Touch device optimizations */
  @media (hover: none) and (pointer: coarse) {
    button, a, input, select, textarea {
      min-height: 48px; /* Larger touch targets */
    }
  }
`;