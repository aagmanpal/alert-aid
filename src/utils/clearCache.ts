/**
 * CACHE CLEAR UTILITY
 * Run this once to clear all cached location and weather data
 */

// Clear all Alert Aid caches
export const clearAllCaches = () => {
  console.log('🗑️ Clearing all Alert Aid caches...');
  
  try {
    // Clear location cache
    localStorage.removeItem('enhanced-location-cache');
    console.log('✅ Location cache cleared');
    
    // Clear any weather caches
    const keysToRemove: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && (key.includes('weather') || key.includes('location') || key.includes('cache'))) {
        keysToRemove.push(key);
      }
    }
    
    keysToRemove.forEach(key => {
      localStorage.removeItem(key);
      console.log(`✅ Cleared: ${key}`);
    });
    
    // Clear session storage
    sessionStorage.clear();
    console.log('✅ Session storage cleared');
    
    console.log('🎉 All caches cleared successfully!');
    console.log('🔄 Reload the page to fetch fresh data');
    
    return true;
  } catch (error) {
    console.error('❌ Failed to clear caches:', error);
    return false;
  }
};

// Auto-clear on import (for development)
if (process.env.NODE_ENV === 'development') {
  const shouldAutoClear = localStorage.getItem('auto-clear-cache');
  if (shouldAutoClear !== 'false') {
    console.log('🔧 Development mode: Auto-clearing caches...');
    clearAllCaches();
    // Prevent auto-clear on next refresh unless manually triggered
    localStorage.setItem('auto-clear-cache', 'false');
  }
}

// Make available globally in console
if (typeof window !== 'undefined') {
  (window as any).clearAllCaches = clearAllCaches;
  console.log('💡 Tip: Run window.clearAllCaches() in console to clear all caches manually');
}
