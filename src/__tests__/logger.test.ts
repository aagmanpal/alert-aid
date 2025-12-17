import logger from '../utils/logger';

// Save original console methods
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;
const originalConsoleError = console.error;

describe('Logger Utility', () => {
  beforeEach(() => {
    // Mock console methods
    console.log = jest.fn();
    console.warn = jest.fn();
    console.error = jest.fn();
  });

  afterEach(() => {
    // Restore original console methods
    console.log = originalConsoleLog;
    console.warn = originalConsoleWarn;
    console.error = originalConsoleError;
  });

  describe('Basic functionality', () => {
    it('logger has all required methods', () => {
      expect(logger.log).toBeDefined();
      expect(logger.warn).toBeDefined();
      expect(logger.error).toBeDefined();
      expect(logger.info).toBeDefined();
      expect(logger.debug).toBeDefined();
    });

    it('logs errors in any environment', () => {
      logger.error('test error');
      expect(console.error).toHaveBeenCalledWith('test error');
    });

    it('accepts multiple arguments', () => {
      logger.log('message', 123, { key: 'value' });
      // In production, log might be silent, but error always works
      logger.error('error', 456, ['array']);
      expect(console.error).toHaveBeenCalled();
    });
  });
});
