/**
 * Production-safe logger utility
 * Logs only appear in development mode to prevent console noise and info leaks in production
 */

const isDevelopment = process.env.NODE_ENV === 'development';

export const logger = {
  /**
   * Development-only informational logs
   * Completely removed in production builds
   */
  log: (...args: any[]): void => {
    if (isDevelopment) {
      console.log(...args);
    }
  },

  /**
   * Development-only warning logs
   * Completely removed in production builds
   */
  warn: (...args: any[]): void => {
    if (isDevelopment) {
      console.warn(...args);
    }
  },

  /**
   * Error logs - always kept in production for critical debugging
   * Use sparingly and ensure no sensitive data is logged
   */
  error: (...args: any[]): void => {
    console.error(...args);
  },

  /**
   * Debug logs - only in development with verbose flag
   * Use for detailed debugging that's too noisy for regular dev
   */
  debug: (...args: any[]): void => {
    if (isDevelopment && process.env.REACT_APP_VERBOSE_LOGGING === 'true') {
      console.debug('[DEBUG]', ...args);
    }
  },

  /**
   * Info logs - kept in production but less verbose than log
   * Use for important application state changes
   */
  info: (...args: any[]): void => {
    if (isDevelopment) {
      console.info(...args);
    }
  }
};

export default logger;
