import { useState, useEffect, useCallback } from 'react';

// Generic API hook types
export interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export interface ApiOptions {
  immediate?: boolean;
  pollingInterval?: number;
  retryCount?: number;
}

// Custom hook for API calls with loading states and error handling
export function useApi<T>(
  apiCall: () => Promise<T>,
  dependencies: any[] = [],
  options: ApiOptions = {}
): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { immediate = true, pollingInterval, retryCount = 3 } = options;

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    let attempts = 0;
    
    while (attempts < retryCount) {
      try {
        const result = await apiCall();
        setData(result);
        setError(null);
        break;
      } catch (err) {
        attempts++;
        const errorMessage = err instanceof Error ? err.message : 'An error occurred';
        
        if (attempts >= retryCount) {
          setError(errorMessage);
          console.error('API call failed after retries:', errorMessage);
        } else {
          console.warn(`API call attempt ${attempts} failed, retrying...`, errorMessage);
          await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
        }
      }
    }
    
    setLoading(false);
  }, [apiCall, retryCount]);

  useEffect(() => {
    if (immediate) {
      fetchData();
    }
  }, [...dependencies, fetchData, immediate]);

  // Set up polling if interval is provided
  useEffect(() => {
    if (pollingInterval && pollingInterval > 0) {
      const interval = setInterval(fetchData, pollingInterval);
      return () => clearInterval(interval);
    }
  }, [fetchData, pollingInterval]);

  return {
    data,
    loading,
    error,
    refetch: fetchData
  };
}

// Hook for real-time data with WebSocket or polling fallback
export function useRealTimeData<T>(
  streamFunction: () => AsyncGenerator<T, void, unknown>,
  fallbackApiCall: () => Promise<T[]>,
  pollingInterval: number = 30000
) {
  const [data, setData] = useState<T[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let cleanup = false;
    
    async function startStream() {
      try {
        setLoading(true);
        setError(null);
        
        // Try WebSocket/streaming first
        try {
          const stream = streamFunction();
          setIsConnected(true);
          setLoading(false);
          
          for await (const item of stream) {
            if (cleanup) break;
            
            setData(prevData => {
              const newData = [...prevData, item];
              // Keep only last 100 items to prevent memory issues
              return newData.slice(-100);
            });
          }
        } catch (streamError) {
          console.warn('Streaming failed, falling back to polling:', streamError);
          setIsConnected(false);
          
          // Fallback to polling
          const pollData = async () => {
            if (cleanup) return;
            
            try {
              const result = await fallbackApiCall();
              setData(result);
              setError(null);
            } catch (pollError) {
              setError(pollError instanceof Error ? pollError.message : 'Polling failed');
            }
          };
          
          // Initial fetch
          await pollData();
          setLoading(false);
          
          // Set up polling
          const interval = setInterval(pollData, pollingInterval);
          return () => clearInterval(interval);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Connection failed');
        setLoading(false);
        setIsConnected(false);
      }
    }

    startStream();
    
    return () => {
      cleanup = true;
    };
  }, [streamFunction, fallbackApiCall, pollingInterval]);

  return {
    data,
    loading,
    error,
    isConnected,
    reconnect: () => {
      setData([]);
      setLoading(true);
      setError(null);
    }
  };
}

// Mutation hook for POST/PUT/DELETE operations
export function useMutation<TData, TVariables = void>(
  mutationFn: (variables: TVariables) => Promise<TData>
) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<TData | null>(null);

  const mutate = useCallback(async (variables: TVariables) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await mutationFn(variables);
      setData(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Mutation failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [mutationFn]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    mutate,
    data,
    loading,
    error,
    reset
  };
}